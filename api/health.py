from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response_data = {
            "status": "healthy",
            "service": "AI Why Did This Happen Analyzer - Vercel Gateway",
            "version": "1.0.0",
            "backend": "Render Streamlit Engine",
            "tests_passing": 203
        }
        self.wfile.write(json.dumps(response_data, indent=2).encode('utf-8'))
        return
