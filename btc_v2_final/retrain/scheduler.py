# retrain/scheduler.py
"""
Auto-retrain scheduler.
Runs weekly (Sunday 02:00 UTC) or when enough new signals accumulate.

Run:
    python retrain/scheduler.py
"""

import sys, os, time, requests, logging, schedule

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import API_URL, MODEL_PATH, SYMBOL, BASE_TF
from storage.db import should_retrain, log_model_version
from models.trainer import train

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def retrain_job(min_new: int = 30):
    log.info("Checking retrain condition...")

    if not should_retrain(min_new_signals=min_new):
        log.info(f"Not enough new signals yet (need {min_new})")
        return

    log.info("Starting retrain...")
    try:
        artifact, val, test = train(
            symbol       = SYMBOL,
            interval     = BASE_TF,
            total        = 3000,
            label_method = "vol_adjusted",
            model_path   = MODEL_PATH,
        )

        log_model_version({
            "version":  artifact["meta"]["version"],
            "cv_auc":   val["mean_auc"],
            "test_auc": test["test_auc"],
            "n_samples": artifact["meta"]["n_samples"],
        })
        log.info(f"Retrain complete — CV AUC: {val['mean_auc']} | Test AUC: {test['test_auc']}")

        # Hot-reload API
        try:
            r = requests.post(f"{API_URL}/reload", timeout=10)
            log.info(f"API reloaded: {r.status_code}")
        except Exception as e:
            log.warning(f"API reload failed (API might not be running): {e}")

    except Exception as e:
        log.error(f"Retrain failed: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    log.info("Retrain scheduler starting...")
    retrain_job()  # run once immediately
    schedule.every().sunday.at("02:00").do(retrain_job)

    while True:
        schedule.run_pending()
        time.sleep(60)
