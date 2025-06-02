from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, Accept')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

if __name__ == '__main__':
    # 切换到前端文件所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 启动前端服务器
    server_address = ('', 3000)
    httpd = HTTPServer(server_address, CORSRequestHandler)
    print('前端服务器运行在 http://localhost:3000')
    httpd.serve_forever() 