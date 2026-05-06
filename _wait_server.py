"""서버가 뜰 때까지 스피너를 출력하는 헬퍼 스크립트."""
import socket
import time
import sys

chars = "|/-\\"
i = 0
while True:
    try:
        s = socket.socket()
        s.settimeout(0.2)
        s.connect(("127.0.0.1", 7860))
        s.close()
        print("\r✨ 준비 완료! 100%      ")
        break
    except Exception:
        i += 1
        p = min(99, i // 2)
        print(f"\r{chars[i % 4]} 엔진 로딩 중... {p}%", end="")
        sys.stdout.flush()
        time.sleep(0.1)
