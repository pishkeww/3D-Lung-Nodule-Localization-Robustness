
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\kinect codes\files\outputs\results.csv")
plt.figure(figsize=(8,5))
plt.hist(df["activation_drift"], bins=20)
plt.xlabel("Activation Drift")
plt.ylabel("Frequency")
plt.title("Distribution of Grad-CAM Drift Under Noise")

plt.tight_layout()

plt.savefig(r"D:\kinect codes\files\drift_histogram.png", dpi=300)

print("Histogramm saved.")

