"""本地 Web 服务：浏览器实时查看微调进度（多个小图，自动刷新）."""
import http.server
import socketserver
import threading
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(".").resolve()
IMG_DIR = ROOT / "output" / "qwen_progress"


def update_plot():
    while True:
        try:
            subprocess.run([sys.executable, "scripts/plot_finetune_progress.py"],
                           cwd=ROOT, capture_output=True)
        except Exception:
            pass
        time.sleep(10)


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            html = f"""<html><head><meta charset="utf-8">
<title>Qwen 微调进度</title>
<script>
setInterval(function(){{
  document.querySelectorAll('img').forEach(function(img){{
    img.src = img.dataset.src.split('?')[0] + '?t=' + Date.now();
  }});
}}, 5000);
</script>
<style>
body {{ background:#111; color:#eee; font-family:sans-serif; text-align:center; }}
h2 {{ margin:20px }}
.imgwrap {{ display:inline-block; width:48%; margin:5px }}
.imgwrap img {{ width:100%; border-radius:8px }}
</style>
</head><body>
<h2>Qwen3-4B QLoRA 微调进度（自动刷新）</h2>
<div class="imgwrap"><img data-src="/img/train_loss.png" src="/img/train_loss.png"></div>
<div class="imgwrap"><img data-src="/img/val_loss.png" src="/img/val_loss.png"></div>
<div class="imgwrap"><img data-src="/img/grad_norm.png" src="/img/grad_norm.png"></div>
<div class="imgwrap"><img data-src="/img/learning_rate.png" src="/img/learning_rate.png"></div>
</body></html>"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        elif self.path.startswith("/img/"):
            name = self.path.split("/")[-1].split("?")[0]
            img = IMG_DIR / name
            if img.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(img.read_bytes())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


PORT = 8899
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"可视化服务: http://localhost:{PORT}")
    threading.Thread(target=update_plot, daemon=True).start()
    httpd.serve_forever()
