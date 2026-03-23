import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set dark background style
plt.style.use('dark_background')

def plot_global_importance(features, importance):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance, y=features, color='#1abc9c')
    plt.title('Global Feature Interpretability')
    plt.xlabel('Relative Influence')
    plt.show()

def plot_radar_chart(categories, values):
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    values += values[:1]
    
    ax.fill(angles, values, color='#1abc9c', alpha=0.25)
    ax.plot(angles, values, color='#1abc9c', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, weight='bold')
    plt.title('CAD Patient Risk Footprint', pad=20)
    plt.show()

def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 5))
    # Using annot_kws to simulate "shadowy" depth
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                annot_kws={"size": 12, "weight": "bold", "color": "white"})
    plt.title('Reconstruction Confusion Matrix')
    plt.show()