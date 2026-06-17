// This runs securely on Vercel's backend, so it CAN read environment variables!
export default async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    // 1. Read the secret environment variable you will set in Vercel
    const RENDER_API_URL = process.env.RENDER_API_URL;

    if (!RENDER_API_URL) {
        return res.status(500).json({ error: 'Server misconfiguration: Missing API URL' });
    }

    try {
        // 2. Forward the exact payload from your game to the Render Brain
        const renderResponse = await fetch(RENDER_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req.body)
        });

        const data = await renderResponse.json();
        
        // 3. Send the AI's move back to your frontend
        return res.status(200).json(data);
        
    } catch (error) {
        console.error("Error communicating with Render:", error);
        return res.status(500).json({ error: 'Failed to fetch from Brain' });
    }
}