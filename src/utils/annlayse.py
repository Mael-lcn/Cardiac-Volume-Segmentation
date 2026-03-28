import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# 1. Chargement et Nettoyage
df = pd.read_csv('../../../output/log/flow/follow.csv')

# 2. ML Stats Extraction
def get_ml_stats(df):
    stats = {
        "Total Steps": df['step'].max(),
        "Final Loss": df['loss'].iloc[-1],
        "Loss Reduction (%)": (1 - df['loss'].iloc[-1] / df['loss'].iloc[0]) * 100,
        "Max Grad Norm": df['grad_norm'].max(),
        "Avg Loss Scale": df['lg_loss_scale'].mean(),
        "Stability (Grad Std)": df['grad_norm'].std()
    }
    return pd.Series(stats)

print("=== ML TRAINING REPORT ===")
print(get_ml_stats(df))
print("==========================")

# 3. Setup Visualisation
sns.set_theme(style="white", palette="muted")
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3)

# --- A. CONVERGENCE (GLOBAL LOSS) ---
ax1 = fig.add_subplot(gs[0, 0])
sns.lineplot(data=df, x='step', y='loss', ax=ax1, color='blue', label='Global Loss', lw=2)
ax1.set_yscale('log')
ax1.set_title("Training Convergence (Log Loss)", fontweight='bold')

# --- B. DISPERSION DES QUANTILES (Q0-Q3) ---
# Analyse la variance entre les segments de données
ax2 = fig.add_subplot(gs[0, 1])
q_cols = ['loss_q0', 'loss_q1', 'loss_q2', 'loss_q3']
for col in q_cols:
    sns.lineplot(data=df, x='step', y=col, ax=ax2, alpha=0.6, label=col.split('_')[1])
ax2.set_yscale('log')
ax2.set_title("Loss Distribution Across Quantiles", fontweight='bold')

# --- C. PRECISION (MSE) ---
ax3 = fig.add_subplot(gs[0, 2])
sns.lineplot(data=df, x='step', y='mse', ax=ax3, color='orange', lw=2)
ax3.set_yscale('log')
ax3.set_title("MSE Trend (Precision)", fontweight='bold')

# --- D. NUMERICAL STABILITY (GRADIENT VS SCALE) ---
ax4 = fig.add_subplot(gs[1, :2])
ax4_twin = ax4.twinx()
ln1 = ax4.plot(df['step'], df['grad_norm'], color='red', label='Grad Norm (Stability)', alpha=0.8)
ln2 = ax4_twin.plot(df['step'], df['lg_loss_scale'], color='green', label='Log Loss Scale', ls='--')
ax4.set_yscale('log')
ax4.set_title("Optimization Stability: Gradient Norm & Mixed Precision Scale", fontweight='bold')
# Regrouper les légendes
lns = ln1 + ln2
ax4.legend(lns, [l.get_label() for l in lns], loc='upper right')

# --- E. GRADIENT DISTRIBUTION (BOXPLOT) ---
# Pour voir si le gradient est bruité ou stable sur la fin
ax5 = fig.add_subplot(gs[1, 2])
sns.boxplot(y=df['grad_norm'], ax=ax5, color='salmon')
ax5.set_yscale('log')
ax5.set_title("Grad Norm Variance", fontweight='bold')

plt.tight_layout()
plt.show()
