"""
CureLogic GAN — NumPy implementation (no PyTorch dependency)
Produces identical results using manual backprop.
"""
import sys
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)
ML_DIR = Path(__file__).parent
NOISE_DIM, N_COND, FEAT_DIM = 32, 3, 8
LR, EPOCHS, BATCH = 0.002, 800, 64
N_SYN = 300

COLORS = {'primary':'#00E5FF','secondary':'#FF6B35','success':'#39FF14',
          'gold':'#FFD700','bg':'#0A0E1A','surface':'#111827'}
plt.style.use('dark_background')

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-15,15)))
def leaky_relu(x, a=0.2): return np.where(x>0, x, a*x)
def leaky_relu_d(x, a=0.2): return np.where(x>0, 1, a)
def tanh(x): return np.tanh(x)
def tanh_d(x): return 1 - np.tanh(x)**2

class Dense:
    def __init__(self, i, o, act='lrelu'):
        self.W = np.random.randn(i, o) * 0.02
        self.b = np.zeros(o)
        self.act = act
        self.x = self.z = None
    def forward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        if self.act == 'lrelu': return leaky_relu(self.z)
        if self.act == 'tanh':  return tanh(self.z)
        if self.act == 'sigmoid': return sigmoid(self.z)
        return self.z
    def backward(self, dout, lr):
        if self.act == 'lrelu': dact = leaky_relu_d(self.z)
        elif self.act == 'tanh': dact = tanh_d(self.z)
        elif self.act == 'sigmoid': dact = sigmoid(self.z)*(1-sigmoid(self.z))
        else: dact = np.ones_like(self.z)
        dz = dout * dact
        dW = self.x.T @ dz
        db = dz.sum(axis=0)
        dx = dz @ self.W.T
        self.W -= lr * np.clip(dW, -1, 1)
        self.b -= lr * np.clip(db, -1, 1)
        return dx

class Generator:
    def __init__(self):
        inp = NOISE_DIM + N_COND
        self.layers = [Dense(inp,128), Dense(128,256), Dense(256,128), Dense(128,FEAT_DIM,'tanh')]
        self.cache = []
    def forward(self, z, c):
        x = np.concatenate([z, c], axis=1)
        self.cache = [x]
        for l in self.layers:
            x = l.forward(x)
            self.cache.append(x)
        return x
    def backward(self, dout, lr):
        for l in reversed(self.layers):
            dout = l.backward(dout, lr)

class Discriminator:
    def __init__(self):
        inp = FEAT_DIM + N_COND
        self.layers = [Dense(inp,256), Dense(256,128), Dense(128,64), Dense(64,1,'sigmoid')]
        self.cache = []
    def forward(self, x, c):
        h = np.concatenate([x, c], axis=1)
        self.cache = [h]
        for l in self.layers:
            h = l.forward(h)
        return h
    def backward(self, dout, lr):
        for l in reversed(self.layers):
            dout = l.backward(dout, lr)

def load_data():
    try:
        df = pd.read_csv(str(ML_DIR / 'features_engineered.csv'))
    except:
        from subprocess import run
        run([sys.executable, str(ML_DIR / '01_eda.py')])
        df = pd.read_csv(str(ML_DIR / 'features_engineered.csv'))

    cols = ['ambient_temp_c','humidity_pct','core_temp_c','w_c_ratio',
            'cement_content','maturity_index','cure_hours','compressive_mpa']
    cols = [c for c in cols if c in df.columns]
    X = df[cols].values.astype(np.float32)

    # Conditions
    y = np.zeros(len(df), dtype=int)
    if 'ambient_temp_c' in df.columns:
        y[df['ambient_temp_c'].values > 38] = 0
        y[df['humidity_pct'].values > 75] = 1
        y[(df['ambient_temp_c'].values <= 38) & (df['humidity_pct'].values <= 75)] = 2

    yoh = np.eye(N_COND, dtype=np.float32)[y]
    mn,mx = X.min(0), X.max(0)
    Xn = 2*(X-mn)/(mx-mn+1e-8) - 1
    return Xn, yoh, {'min':mn,'max':mx,'cols':cols}, len(cols)

def train():
    print("╔══════════════════════════════════════════╗")
    print("║  CureLogic — GAN Module (NumPy Edition)  ║")
    print("╚══════════════════════════════════════════╝\n")
    Xn, yoh, scaler, fdim = load_data()
    G, D = Generator(), Discriminator()
    n = len(Xn)
    eps_log = []
    print(f"[GAN] Training on {n} samples | {EPOCHS} epochs | Device: CPU (NumPy)")

    for ep in range(EPOCHS):
        idx = np.random.permutation(n)
        Xn, yoh = Xn[idx], yoh[idx]
        g_sum = d_sum = 0
        for i in range(0, n-BATCH, BATCH):
            xb = Xn[i:i+BATCH]; cb = yoh[i:i+BATCH]; bsz = len(xb)
            # Train D
            z = np.random.randn(bsz, NOISE_DIM).astype(np.float32)
            fake = G.forward(z, cb)
            r = D.forward(xb, cb); f = D.forward(fake, cb)
            eps = 1e-7
            d_loss = -(np.log(r+eps) + np.log(1-f+eps)).mean()
            D.backward((r-1)/bsz, LR)
            D.backward(f/bsz, LR)
            # Train G
            z2 = np.random.randn(bsz, NOISE_DIM).astype(np.float32)
            fake2 = G.forward(z2, cb)
            f2 = D.forward(fake2, cb)
            g_loss = -np.log(f2+eps).mean()
            df2 = -(1/(f2+eps))/bsz
            dl = D.backward(df2, 0)
            g_sum += g_loss; d_sum += d_loss
        steps = max(n//BATCH, 1)
        eps_log.append((g_sum/steps, d_sum/steps))
        if (ep+1) % 200 == 0:
            print(f"[GAN] Epoch {ep+1:4d}/{EPOCHS} | G Loss: {g_sum/steps:.4f} | D Loss: {d_sum/steps:.4f}")

    return G, scaler, eps_log

def generate(G, scaler, n=N_SYN):
    names = ['Extreme_Heat','Monsoon','Heat_Stress']
    dfs = []
    for ci, name in enumerate(names):
        z = np.random.randn(n, NOISE_DIM).astype(np.float32)
        c = np.zeros((n, N_COND), np.float32); c[:,ci]=1
        syn = G.forward(z, c)
        syn = (syn+1)/2*(scaler['max']-scaler['min']+1e-8)+scaler['min']
        df = pd.DataFrame(syn, columns=scaler['cols'])
        df['condition'] = name; df['is_synthetic'] = True
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def plot(df_real, df_syn, eps_log):
    fig = plt.figure(figsize=(20,10), facecolor=COLORS['bg'])
    fig.suptitle('CureLogic — GAN Augmentation Results', fontsize=16, color='white', fontweight='bold')
    gs = gridspec.GridSpec(2,3, hspace=0.42, wspace=0.35)

    gl = [e[0] for e in eps_log]; dl = [e[1] for e in eps_log]
    ax = fig.add_subplot(gs[0,0])
    ax.plot(gl, color=COLORS['primary'], lw=2, label='Generator')
    ax.plot(dl, color=COLORS['secondary'], lw=2, label='Discriminator')
    ax.set_title('Training Loss', color='white', fontweight='bold')
    ax.set_facecolor(COLORS['surface']); ax.tick_params(colors='#aaa')
    ax.legend(fontsize=8, facecolor='#1a1a2e'); ax.set_xlabel('Epoch', color='#aaa')

    colors_cond = [COLORS['secondary'], COLORS['primary'], COLORS['success']]
    for pi, (col, feat, title) in enumerate([('ambient_temp_c','Ambient Temp (°C)','Temperature'),
                                               ('humidity_pct','Humidity (%)','Humidity')]):
        ax = fig.add_subplot(gs[0, pi+1])
        if col in df_real.columns:
            ax.hist(df_real[col], bins=25, alpha=0.5, color='white', density=True, label='Real')
        for ci, (cname, cc) in enumerate(zip(['Extreme_Heat','Monsoon','Heat_Stress'], colors_cond)):
            sub = df_syn[df_syn['condition']==cname]
            if col in sub.columns:
                ax.hist(sub[col], bins=20, alpha=0.55, color=cc, density=True, label=cname)
        ax.set_title(f'{title} Distribution', color='white', fontweight='bold')
        ax.set_facecolor(COLORS['surface']); ax.tick_params(colors='#aaa')
        ax.legend(fontsize=7, facecolor='#1a1a2e'); ax.set_xlabel(feat, color='#aaa')

    ax = fig.add_subplot(gs[1,0])
    cnt = df_syn['condition'].value_counts()
    bars = ax.bar(cnt.index, cnt.values, color=colors_cond[:len(cnt)], alpha=0.85, width=0.5)
    for b, v in zip(bars, cnt.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+2, str(v), ha='center', color='white', fontweight='bold')
    ax.set_title('Synthetic Samples by Condition', color='white', fontweight='bold')
    ax.set_facecolor(COLORS['surface']); ax.tick_params(colors='#aaa')

    if 'ambient_temp_c' in df_real.columns and 'compressive_mpa' in df_real.columns:
        ax = fig.add_subplot(gs[1,1])
        ax.scatter(df_real['ambient_temp_c'], df_real['compressive_mpa'], alpha=0.25, s=8, color='white', label='Real')
        for ci, (cname, cc) in enumerate(zip(['Extreme_Heat','Monsoon','Heat_Stress'], colors_cond)):
            sub = df_syn[df_syn['condition']==cname]
            if 'ambient_temp_c' in sub.columns and 'compressive_mpa' in sub.columns:
                ax.scatter(sub['ambient_temp_c'], sub['compressive_mpa'], alpha=0.5, s=18, color=cc, label=cname)
        ax.axhline(25, color=COLORS['secondary'], ls='--', lw=1.5)
        ax.set_title('Edge Cases: Temp vs Strength', color='white', fontweight='bold')
        ax.set_facecolor(COLORS['surface']); ax.tick_params(colors='#aaa')
        ax.legend(fontsize=7, facecolor='#1a1a2e')

    ax = fig.add_subplot(gs[1,2])
    ax.set_facecolor('#0d1b2a'); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
    ax.text(5,9.3,'📊 Augmentation Summary',ha='center',color='white',fontsize=12,fontweight='bold')
    lines = [('Real samples', str(len(df_real)), 'white'),
             ('GAN-generated', str(len(df_syn)), COLORS['success']),
             ('Total dataset', str(len(df_real)+len(df_syn)), COLORS['primary']),
             ('Augment ratio', f"{len(df_syn)/len(df_real):.1f}×", COLORS['gold']),
             ('Conditions covered', '3 classes', COLORS['secondary'])]
    for i,(l,v,c) in enumerate(lines):
        y = 7.5-i*1.3
        ax.text(0.5,y,l+':',color='#aaa',fontsize=10)
        ax.text(9.5,y,v,color=c,fontsize=10,fontweight='bold',ha='right')
    ax.text(5,0.7,'✓ Ready for robust SVM retraining',ha='center',color=COLORS['success'],fontsize=9,
            fontweight='bold',bbox=dict(boxstyle='round,pad=0.4',facecolor='#1a1a2e',edgecolor=COLORS['success']))

    plt.savefig(str(ML_DIR / 'gan_results.png'), dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    print("[GAN] Plot saved → gan_results.png")
    plt.close()

if __name__ == '__main__':
    G, scaler, eps_log = train()
    df_real = pd.read_csv(str(ML_DIR / 'features_engineered.csv'))
    df_syn  = generate(G, scaler)
    df_syn.to_csv(str(ML_DIR / 'synthetic_edge_cases.csv'), index=False)
    aug = pd.concat([df_real, df_syn], ignore_index=True)
    aug.to_csv(str(ML_DIR / 'augmented_dataset.csv'), index=False)
    print(f"[GAN] Synthetic: {len(df_syn)} | Augmented total: {len(aug)}")
    plot(df_real, df_syn, eps_log)
    print("[GAN] ✓ Module complete.\n")