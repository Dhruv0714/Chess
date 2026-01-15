from flask import Flask, request, jsonify
from flask_cors import CORS
import chess
import math

app = Flask(__name__)
CORS(app)  # Allows your local HTML file to talk to this server

# --- Coordinate Helpers ---
# Your board uses "x_y" (e.g., "1_1"). Chess uses "a1".
def map_to_algebraic(pos):
    x, y = map(int, pos.split('_'))
    col = chr(96 + x)  # 1->a, 2->b...
    return f"{col}{y}"

def map_from_algebraic(square_name):
    col_char = square_name[0]
    row_char = square_name[1]
    x = ord(col_char) - 96
    y = int(row_char)
    return f"{x}_{y}"

# --- Board Reconstruction ---
def reconstruct_board(pieces_data):
    board = chess.Board(None) # Start with empty board
    
    # Map your piece names to python-chess types
    type_map = {
        'king': chess.KING, 'queen': chess.QUEEN, 'rook': chess.ROOK,
        'bishop': chess.BISHOP, 'knight': chess.KNIGHT, 'pawn': chess.PAWN
    }
    
    for key, piece in pieces_data.items():
        if not piece['captured']:
            # Determine color and type
            color = chess.WHITE if piece['type'].startswith('w_') else chess.BLACK
            p_type = piece['type'].split('_')[1]
            
            # Place the piece
            alg_pos = map_to_algebraic(piece['position'])
            square = chess.parse_square(alg_pos)
            board.set_piece_at(square, chess.Piece(type_map[p_type], color))
            
    # Set Castling Rights (Important for optimal play)
    board.castling_rights = 0
    # White
    if not pieces_data['w_king']['moved']:
        if not pieces_data['w_rook2']['moved']: board.castling_rights |= chess.BB_H1
        if not pieces_data['w_rook1']['moved']: board.castling_rights |= chess.BB_A1
    # Black
    if not pieces_data['b_king']['moved']:
        if not pieces_data['b_rook2']['moved']: board.castling_rights |= chess.BB_H8
        if not pieces_data['b_rook1']['moved']: board.castling_rights |= chess.BB_A8

    # It's Black's turn
    board.turn = chess.BLACK
    return board

# --- The AI Brain (Minimax with Alpha-Beta Pruning) ---
def evaluate_board(board):
    if board.is_checkmate():
        if board.turn: return -9999 # Black wins
        else: return 9999 # White wins
        
    evaluation = 0
    values = { chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0 }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            val = values[piece.piece_type]
            if piece.color == chess.WHITE: evaluation -= val
            else: evaluation += val
    return evaluation

def minimax(board, depth, alpha, beta, maximizing):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    moves = list(board.legal_moves)
    if maximizing: # AI (Black)
        max_eval = -math.inf
        for move in moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha: break
        return max_eval
    else: # Player (White)
        min_eval = math.inf
        for move in moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha: break
        return min_eval

@app.route('/', defaults={'path': ''}, methods=['POST'])
@app.route('/<path:path>', methods=['POST'])
def get_move(path):
    data = request.json
    try:
        board = reconstruct_board(data['pieces'])
        
        # Depth 3 is a good balance of speed vs intelligence
        best_move = None
        best_value = -math.inf
        
        for move in board.legal_moves:
            board.push(move)
            val = minimax(board, 2, -math.inf, math.inf, False)
            board.pop()
            if val > best_value:
                best_value = val
                best_move = move
        
        if best_move:
            src = map_from_algebraic(chess.square_name(best_move.from_square))
            dst = map_from_algebraic(chess.square_name(best_move.to_square))
            return jsonify({"from": src, "to": dst})
        
        return jsonify({"error": "Game Over or No Move"})
        
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run()