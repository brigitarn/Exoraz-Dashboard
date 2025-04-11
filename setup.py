import subprocess

print("Running fetch_comments.py...")
subprocess.run(["python", "fetch_comments.py"], check=True)

print("Training model...")
subprocess.run(["python", "train_model.py"], check=True)

print("Running fetch_video_stats.py...")
subprocess.run(["python", "fetch_video_stats.py"], check=True)

print("✅ Setup completed.")
