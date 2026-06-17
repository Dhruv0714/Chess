from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np

app = Flask(__name__)
CORS(app) # Allows your JS frontend to talk to this backend

# --- 1. REBUILD THE BRAIN ARCHITECTURE ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + x)

class ChessNet(nn.Module):
    def __init__(self, num_blocks=10, channels=128):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(18, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU()
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(), nn.Linear(2 * 8 * 8, 4096)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1), nn.ReLU(), nn.Flatten(),
            nn.Linear(1 * 8 * 8, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh()
        )
    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.res_blocks: x = block(x)
        return self.policy_head(x), self.value_head(x)

# Load the model globally so it's ready when requests come in
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessNet().to(device)
model.load_state_dict(torch.load("gm_model_v1.pth", map_location=device))
model.eval()
print("✅ Brain loaded and server is ready!")

# --- 2. TRANSLATION HELPERS ---
def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square

def extract_features(board: chess.Board):
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    us = board.turn 
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece: continue
        rank = chess.square_rank(square) if us == chess.WHITE else 7 - chess.square_rank(square)
        file = chess.square_file(square) if us == chess.WHITE else 7 - chess.square_file(square)
        is_ours = (piece.color == us)
        plane_idx = (piece.piece_type - 1) + (0 if is_ours else 6)
        planes[plane_idx, rank, file] = 1.0
    if us == chess.WHITE: planes[12, :, :] = 1.0
    return planes

def js_to_board(js_pieces):
    """Translates your custom JS layout into a python-chess board"""
    board = chess.Board(None) # Start with empty board
    
    piece_map = {
        'pawn': chess.PAWN, 'knight': chess.KNIGHT, 'bishop': chess.BISHOP,
        'rook': chess.ROOK, 'queen': chess.QUEEN, 'king': chess.KING
    }
    
    for key, piece_data in js_pieces.items():
        if piece_data.get('captured'):
            continue
            
        # Parse JS "5_1" into file(4) and rank(0)
        x_str, y_str = piece_data['position'].split('_')
        file_idx = int(x_str) - 1
        rank_idx = int(y_str) - 1
        square = chess.square(file_idx, rank_idx)
        
        # Parse color and type from "w_king"
        color = chess.WHITE if piece_data['type'].startswith('w') else chess.BLACK
        piece_type = piece_map[piece_data['type'].split('_')[1]]
        
        board.set_piece_at(square, chess.Piece(piece_type, color))
    
    # Since the frontend calls AI on Black's turn
    board.turn = chess.BLACK
    return board

def square_to_js(square):
    """Translates python-chess square back to JS 'x_y' format"""
    x = chess.square_file(square) + 1
    y = chess.square_rank(square) + 1
    return f"{x}_{y}"

# --- 3. THE API ENDPOINT ---
@app.route('/api/brain', methods=['POST'])
def ai_move():
    data = request.json
    if not data or 'pieces' not in data:
        return jsonify({"error": "Invalid payload"}), 400
        
    # Translate JS to python-chess
    board = js_to_board(data['pieces'])
    
    # Check if game is already over
    if board.is_checkmate():
        return jsonify({"winner": "white"}) # Black is checkmated
        
    # Extract features for Neural Net
    features = extract_features(board)
    tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    
    # Ask the Brain!
    with torch.no_grad():
        policy, value = model(tensor)
        
    # Find the highest scored legal move
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return jsonify({"error": "No legal moves available"}), 400
        
    best_move = max(legal_moves, key=lambda m: policy[0, move_to_index(m)].item())
    
    # Translate python-chess move back to JS coordinates
    from_js = square_to_js(best_move.from_square)
    to_js = square_to_js(best_move.to_square)
    
    print(f"🤖 AI chose: {best_move} (Translating to JS: {from_js} -> {to_js})")
    
    return jsonify({
        "from": from_js,
        "to": to_js
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)