"""
╔══════════════════════════════════════════════════════════════╗
║         CureLogic — Module 1: Exploratory Data Analysis      ║
║         L&T CreaTech Hackathon | Problem Statement 1         ║
╚══════════════════════════════════════════════════════════════╝

Merges historical concrete curing sensor data with live OpenWeather
API data, performs comprehensive EDA, and outputs feature-engineered
dataset ready for SVM training.

Key findings visualized:
  • Temperature vs Strength gain curves
  • Maturity index distribution by season
  • Weather correlation heatmap
  • Cost-per-MPa efficiency scatter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import requests
import json
import os
from pathlib import Path
import warnings
from datetime import datetime, timedelta
from scipy import stats

warnings.filterwarnings('ignore')

# ─── Paths (cross-platform: works on Windows, Mac, Linux) ────
ML_DIR = Path(__file__).parent

# ─── Configuration ────────────────────────────────────────────
OPENWEATHER_API_KEY = os.getenv("OW_API_KEY", "YOUR_API_KEY_HERE")
CITY               = "Pune,IN"          # L&T precast yard location
WATER_CURE_COST    = 12.5               # INR/hr
MOULD_COST         = 450.0             # INR/hr mould occupation cost
TARGET_MPa         = 25.0

plt.style.use('dark_background')
COLORS = {
    'primary':    '#00E5FF',
    'secondary':  '#FF6B35',
    'success':    '#39FF14',
    'warning':    '#FFD700',
    'danger':     '#FF073A',
    'bg':         '#0A0E1A',
    'surface':    '#111827',
}

# ─────────────────────────────────────────────────────────────
# 1. GENERATE SYNTHETIC HISTORICAL DATA (fallback if no CSV)
# ─────────────────────────────────────────────────────────────
def generate_curing_dataset(n_batches: int = 500) -> pd.DataFrame:
    """
    Simulates realistic concrete curing records for M30 precast elements.
    Each row = one hourly reading from one batch pour.
    """
    print("[EDA] Generating synthetic curing dataset...")
    np.random.seed(42)
    records = []

    seasons = {
        'Summer':  {'temp_mean': 38, 'temp_std': 4,  'humidity_mean': 35, 'humidity_std': 10},
        'Monsoon': {'temp_mean': 30, 'temp_std': 3,  'humidity_mean': 85, 'humidity_std': 8},
        'Winter':  {'temp_mean': 18, 'temp_std': 5,  'humidity_mean': 55, 'humidity_std': 12},
        'Spring':  {'temp_mean': 28, 'temp_std': 5,  'humidity_mean': 50, 'humidity_std': 10},
    }

    for batch_id in range(n_batches):
        season = np.random.choice(list(seasons.keys()))
        s = seasons[season]

        ambient_temp  = np.clip(np.random.normal(s['temp_mean'], s['temp_std']), 10, 50)
        humidity      = np.clip(np.random.normal(s['humidity_mean'], s['humidity_std']), 20, 100)
        w_c_ratio     = np.random.uniform(0.40, 0.55)
        cement_content= np.random.uniform(350, 450)    # kg/m³
        curing_method = np.random.choice(['Wet', 'Steam', 'Membrane'], p=[0.5, 0.3, 0.2])

        # Internal temperature (core heats up from hydration)
        hydration_heat = cement_content * 0.04 * np.random.uniform(0.8, 1.2)
        core_temp      = ambient_temp + hydration_heat * np.random.uniform(0.5, 1.5)
        core_temp      = np.clip(core_temp, ambient_temp, ambient_temp + 30)

        # Maturity (Nurse-Saul): M = Σ(T - T0)·Δt  |  T0 = -10°C, Δt = 1hr
        T0         = -10.0
        cure_hours = np.random.uniform(12, 48)
        # Simplified: average temp over cure period
        avg_temp   = (core_temp + ambient_temp) / 2.0
        maturity   = (avg_temp - T0) * cure_hours

        # Strength (Logarithmic model + noise)
        raw_strength  = -12.4 + 7.8 * np.log(max(maturity, 1))
        noise         = np.random.normal(0, 1.2)

        # Method bonus
        method_bonus  = {'Steam': 3.5, 'Wet': 0.0, 'Membrane': -1.2}[curing_method]
        strength      = np.clip(raw_strength + method_bonus + noise, 5, 58)

        # Cost calculation
        method_cost_hr = {'Wet': WATER_CURE_COST, 'Steam': 22.0, 'Membrane': 8.0}[curing_method]
        total_cost     = (method_cost_hr + MOULD_COST) * cure_hours
        cost_per_mpa   = total_cost / max(strength, 1)

        # Demoulding decision
        demoulded      = strength >= TARGET_MPa

        records.append({
            'batch_id':         batch_id,
            'season':           season,
            'ambient_temp_c':   round(ambient_temp, 2),
            'humidity_pct':     round(humidity, 1),
            'core_temp_c':      round(core_temp, 2),
            'w_c_ratio':        round(w_c_ratio, 3),
            'cement_content':   round(cement_content, 1),
            'curing_method':    curing_method,
            'cure_hours':       round(cure_hours, 2),
            'maturity_index':   round(maturity, 1),
            'compressive_mpa':  round(strength, 2),
            'total_cost_inr':   round(total_cost, 2),
            'cost_per_mpa':     round(cost_per_mpa, 2),
            'demoulded':        int(demoulded),
        })

    df = pd.DataFrame(records)
    print(f"[EDA] Dataset created: {df.shape[0]} records × {df.shape[1]} features\n")
    return df


# ─────────────────────────────────────────────────────────────
# 2. FETCH LIVE WEATHER DATA
# ─────────────────────────────────────────────────────────────
def fetch_live_weather() -> dict:
    """Fetches current weather from OpenWeatherMap API."""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            weather = {
                'temp':       data['main']['temp'],
                'humidity':   data['main']['humidity'],
                'feels_like': data['main']['feels_like'],
                'wind_speed': data['wind']['speed'],
                'condition':  data['weather'][0]['description'],
            }
            print(f"[Weather API] Live: {weather['temp']}°C, {weather['humidity']}% RH, {weather['condition']}")
            return weather
        else:
            raise Exception(f"API error {r.status_code}")
    except Exception as e:
        print(f"[Weather API] Offline — using fallback values. ({e})")
        return {'temp': 32.0, 'humidity': 60, 'feels_like': 34.0, 'wind_speed': 3.2, 'condition': 'clear sky'}


# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived features critical for SVM training."""
    print("[EDA] Engineering features...")

    # Thermal efficiency: how much heat the concrete absorbs vs ambient
    df['thermal_delta']    = df['core_temp_c'] - df['ambient_temp_c']

    # Normalised maturity (log scale useful for SVM RBF kernel)
    df['log_maturity']     = np.log1p(df['maturity_index'])

    # Season encoding
    season_map = {'Summer': 0, 'Monsoon': 1, 'Winter': 2, 'Spring': 3}
    df['season_enc']       = df['season'].map(season_map)

    # Method encoding
    method_map = {'Wet': 0, 'Steam': 1, 'Membrane': 2}
    df['method_enc']       = df['curing_method'].map(method_map)

    # Cost efficiency flag: cost-effective if below median cost-per-MPa
    median_cpm             = df['cost_per_mpa'].median()
    df['cost_efficient']   = (df['cost_per_mpa'] < median_cpm).astype(int)

    # Maturity rate (°C·hr per cure hour)
    df['maturity_rate']    = df['maturity_index'] / df['cure_hours']

    # Humidity-temperature index (heat stress factor)
    df['heat_stress']      = (df['ambient_temp_c'] * df['humidity_pct']) / 100.0

    print(f"[EDA] Features after engineering: {df.shape[1]}\n")
    return df


# ─────────────────────────────────────────────────────────────
# 4. EDA VISUALIZATIONS
# ─────────────────────────────────────────────────────────────
def plot_eda(df: pd.DataFrame, weather: dict):
    """Generates a comprehensive 8-panel EDA dashboard figure."""
    print("[EDA] Generating visualizations...")

    fig = plt.figure(figsize=(22, 16), facecolor=COLORS['bg'])
    fig.suptitle(
        'CureLogic — Exploratory Data Analysis Dashboard\nL&T CreaTech | AI Cycle Time Optimization',
        fontsize=18, color='white', fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Strength vs Maturity ───────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    for method, color in [('Wet', COLORS['primary']), ('Steam', COLORS['secondary']), ('Membrane', COLORS['warning'])]:
        sub = df[df['curing_method'] == method]
        ax1.scatter(sub['maturity_index'], sub['compressive_mpa'],
                    alpha=0.4, s=18, color=color, label=method)

    # Fit curve
    x_fit   = np.linspace(df['maturity_index'].min(), df['maturity_index'].max(), 300)
    a, b    = -12.4, 7.8
    y_fit   = np.clip(a + b * np.log(x_fit), 0, 60)
    ax1.plot(x_fit, y_fit, color=COLORS['success'], lw=2.5, label='Nurse-Saul Model', zorder=5)
    ax1.axhline(y=TARGET_MPa, color=COLORS['danger'], lw=1.5, ls='--', label=f'Target {TARGET_MPa} MPa')
    ax1.set_xlabel('Maturity Index (°C·hr)', color='#aaa')
    ax1.set_ylabel('Compressive Strength (MPa)', color='#aaa')
    ax1.set_title('Strength vs Maturity Index', color='white', fontweight='bold')
    ax1.legend(fontsize=8, facecolor='#1a1a2e')
    ax1.set_facecolor(COLORS['surface'])
    ax1.tick_params(colors='#aaa')

    # ── Panel 2: Cost Distribution by Curing Method ─────────
    ax2 = fig.add_subplot(gs[0, 2:])
    method_colors = [COLORS['primary'], COLORS['secondary'], COLORS['warning']]
    methods = df['curing_method'].unique()
    data_to_plot = [df[df['curing_method'] == m]['cost_per_mpa'].values for m in methods]

    bp = ax2.boxplot(data_to_plot, patch_artist=True, labels=methods,
                     medianprops=dict(color='white', linewidth=2))
    for patch, color in zip(bp['boxes'], method_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Cost per MPa (INR)', color='#aaa')
    ax2.set_title('Cost Efficiency by Curing Method', color='white', fontweight='bold')
    ax2.set_facecolor(COLORS['surface'])
    ax2.tick_params(colors='#aaa')

    # ── Panel 3: Seasonal Cure Time Distribution ─────────────
    ax3 = fig.add_subplot(gs[1, :2])
    season_order = ['Summer', 'Monsoon', 'Spring', 'Winter']
    season_colors = [COLORS['secondary'], COLORS['primary'], COLORS['success'], COLORS['warning']]
    for i, (season, color) in enumerate(zip(season_order, season_colors)):
        sub = df[df['season'] == season]['cure_hours']
        ax3.hist(sub, bins=20, alpha=0.65, color=color, label=season, density=True)
    ax3.set_xlabel('Cure Hours to Target Strength', color='#aaa')
    ax3.set_ylabel('Density', color='#aaa')
    ax3.set_title('Cure Time Distribution by Season', color='white', fontweight='bold')
    ax3.legend(fontsize=8, facecolor='#1a1a2e')
    ax3.set_facecolor(COLORS['surface'])
    ax3.tick_params(colors='#aaa')

    # ── Panel 4: Correlation Heatmap ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 2:])
    numeric_cols = ['ambient_temp_c', 'humidity_pct', 'core_temp_c',
                    'maturity_index', 'cure_hours', 'compressive_mpa',
                    'total_cost_inr', 'heat_stress']
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, ax=ax4,
                annot=True, fmt='.2f', annot_kws={'size': 7},
                linewidths=0.5, linecolor='#333')
    ax4.set_title('Feature Correlation Matrix', color='white', fontweight='bold')
    ax4.tick_params(colors='#aaa', labelsize=7)

    # ── Panel 5: Cost Saving Opportunity ─────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    seasons  = df.groupby('season')['cure_hours'].mean()
    baseline = df[df['curing_method'] == 'Wet']['cure_hours'].mean()
    savings  = df.groupby('curing_method')['cure_hours'].mean().apply(
        lambda x: (baseline - x) * MOULD_COST
    )
    bars = ax5.bar(savings.index, savings.values,
                   color=[COLORS['primary'], COLORS['secondary'], COLORS['success']],
                   alpha=0.85, width=0.5)
    for bar, val in zip(bars, savings.values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'INR {val:.0f}', ha='center', color='white', fontsize=9, fontweight='bold')
    ax5.set_ylabel('Avg. Cost Saved vs Wet Cure (INR)', color='#aaa')
    ax5.set_title('💰 Potential Savings by Curing Method', color='white', fontweight='bold')
    ax5.set_facecolor(COLORS['surface'])
    ax5.tick_params(colors='#aaa')

    # ── Panel 6: Live Weather Tile ────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2:])
    ax6.set_facecolor('#0d1b2a')
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    ax6.axis('off')

    ax6.text(5, 9.2, '🌡  Live Weather — Pune (L&T Yard)',
             ha='center', color='white', fontsize=11, fontweight='bold')

    live_data = [
        ('Temperature',  f"{weather['temp']} °C",        COLORS['secondary']),
        ('Humidity',     f"{weather['humidity']} %",     COLORS['primary']),
        ('Feels Like',   f"{weather['feels_like']} °C",  COLORS['warning']),
        ('Wind Speed',   f"{weather['wind_speed']} m/s", COLORS['success']),
        ('Condition',    weather['condition'].title(),   'white'),
    ]

    for i, (label, value, color) in enumerate(live_data):
        y = 7.5 - i * 1.4
        ax6.text(1, y, label + ':',  color='#aaa',  fontsize=10)
        ax6.text(9, y, value,        color=color,   fontsize=10, fontweight='bold', ha='right')

    # Curing recommendation based on live weather
    rec_color = COLORS['success']
    if weather['temp'] > 38:
        rec = "⚠  HIGH HEAT — Increase curing frequency"
        rec_color = COLORS['danger']
    elif weather['humidity'] > 80:
        rec = "✓  HIGH HUMIDITY — Reduce water usage 15%"
        rec_color = COLORS['warning']
    else:
        rec = "✓  OPTIMAL — Standard curing schedule"

    ax6.text(5, 0.6, rec, ha='center', color=rec_color, fontsize=10,
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e', edgecolor=rec_color))

    plt.savefig(str(ML_DIR / 'eda_dashboard.png'),
                dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    print("[EDA] Dashboard saved → ml/eda_dashboard.png")
    plt.close()


# ─────────────────────────────────────────────────────────────
# 5. STATISTICAL SUMMARY
# ─────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    print("\n" + "═"*60)
    print("  CureLogic EDA — Key Findings")
    print("═"*60)
    print(f"  Total batches analysed : {len(df)}")
    print(f"  Avg cure time          : {df['cure_hours'].mean():.1f} hrs")
    print(f"  Fastest cure (Steam)   : {df[df['curing_method']=='Steam']['cure_hours'].min():.1f} hrs")
    print(f"  Avg strength achieved  : {df['compressive_mpa'].mean():.1f} MPa")
    print(f"  Demoulding success rate: {df['demoulded'].mean()*100:.1f}%")
    print(f"  Avg cost per batch     : INR {df['total_cost_inr'].mean():.0f}")
    print(f"  Best cost/MPa method   : {df.groupby('curing_method')['cost_per_mpa'].mean().idxmin()}")

    steam_save = (df[df['curing_method']=='Wet']['cure_hours'].mean() -
                  df[df['curing_method']=='Steam']['cure_hours'].mean())
    cost_save  = steam_save * MOULD_COST
    print(f"\n  💰 Steam vs Wet cure saves {steam_save:.1f} hrs avg → INR {cost_save:.0f}/batch")
    print(f"  📈 Top corr with strength: maturity_index ({df['maturity_index'].corr(df['compressive_mpa']):.3f})")
    print("═"*60 + "\n")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════╗")
    print("║  CureLogic — EDA Module | L&T CreaTech 2024  ║")
    print("╚══════════════════════════════════════════════╝\n")

    # Load or generate data
    DATA_PATH = str(ML_DIR / "curing_data.csv")
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print(f"[EDA] Loaded existing dataset: {DATA_PATH}")
    else:
        df = generate_curing_dataset(n_batches=500)

    # Weather
    weather = fetch_live_weather()

    # Feature engineering
    df = engineer_features(df)

    # Save engineered dataset
    df.to_csv(str(ML_DIR / "features_engineered.csv"), index=False)
    print(f"[EDA] Engineered features saved → {ML_DIR / 'features_engineered.csv'}")

    # Visualize
    plot_eda(df, weather)

    # Summary
    print_summary(df)

    print("[EDA] ✓ Module complete. Feed features_engineered.csv into 02_svm_model.py\n")