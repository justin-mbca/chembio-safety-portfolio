# ChemBio SafeGuard Configuration

# API Configuration
API_HOST = "localhost"
API_PORT = 3000
FRONTEND_PORT = 3001

# Database Configuration
DATABASE_UPDATE_INTERVAL = 86400  # 24 hours in seconds
DATABASE_CACHE_SIZE = 1000

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "logs/chembio_safety.log"

# Model Configuration
MODEL_CONFIDENCE_THRESHOLD = 0.7
MAX_INPUT_LENGTH = 1000

# Security Configuration
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 3600  # 1 hour in seconds
