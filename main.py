import logging
from app import app

# Setup logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # Use debug=False so the process (and loaded models) are not reloaded automatically.
    app.run(host="0.0.0.0", port=5000, debug=False)
