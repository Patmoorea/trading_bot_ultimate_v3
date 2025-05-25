1i\
import logging\
import json  # Standard library first\

/^class RealTimeBot:/i\
# ============ AJOUT SAFE_LOG ============ #\
def safe_log(message: str, level: str = "info"):\
    """Logging sécurisé avec format lazy"""\
    if level == "info":\
        logging.info("%s", message)\
    elif level == "warning":\
        logging.warning("%s", message)\
    elif level == "error":\
        logging.error("%s", message)\
# ============ FIN AJOUT ============ #\

s/logger\.info(\([^)]*\))/safe_log(\1)/g
s/logger\.warning(\([^)]*\))/safe_log(\1, "warning")/g
s/logger\.error(\([^)]*\))/safe_log(\1, "error")/g
