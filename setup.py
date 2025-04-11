import pandas as pd
import fetch_comments
import train_model
import fetch_video_stats

def run_setup():
    print("Running fetch_comments.py...")
    fetch_comments.main()

    print("Training model...")
    train_model.main()

    print("Running fetch_video_stats.py...")
    fetch_video_stats.main()

    print("✅ Setup completed.")