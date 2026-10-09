"""Builds report/report.html from results/results.json + figures/, then prints it to HW2_Report.pdf.

Usage:  python report/make_report.py [GITHUB_URL]
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_URL = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/sh-nikhil/CS6220_Linear-Regression-and-Regression-Trees.git"
R = json.load(open(os.path.join(ROOT, "results", "results.json")))

FUNCS = ["linear", "cubic"]
EPS = ["0.01", "0.1"]
SCHEMAS = ["(x,y)", "(x,x^2,x^3,y)"]
SCH_HTML = {"(x,y)": "(x, y)", "(x,x^2,x^3,y)": "(x, x², x³, y)"}
FUNC_HTML = {"linear": "Linear 2x+1", "cubic": "Cubic 0.5x³−x²+x"}


def f(v, d=5):
    return f"{v:.{d}f}"


def poly(beta):
    s = f"{beta[0]:.4f}"
    for i, b in enumerate(beta[1:], 1):
        s += f" {'+' if b >= 0 else '−'} {abs(b):.4f}x" + ("" if i == 1 else {2: "²", 3: "³"}[i])
    return s


# ---------- tables ----------
def lin_table(schema):
    rows = ""
    for fn in FUNCS:
        for e in EPS:
            r = R["linear"][f"{schema} | {fn} | {e}"]
            rows += f"<tr><td>{FUNC_HTML[fn]}</td><td>{e}</td><td class=m>{poly(r['beta'])}</td><td>{f(r['train_rmse'])}</td><td>{f(r['test_rmse'])}</td></tr>"
    return f"<table><tr><th>True f</th><th>ε</th><th>Learned model f̂(x)</th><th>Train RMSE</th><th>Test RMSE</th></tr>{rows}</table>"


def extrap_table():
    rows = ""
    for s in SCHEMAS:
        for fn in FUNCS:
            for e in EPS:
                r = R["extrapolation"][f"{s} | {fn} | {e}"]
                rows += f"<tr><td>{SCH_HTML[s]}</td><td>{FUNC_HTML[fn]}</td><td>{e}</td><td>{r['true']:.0f}</td><td>{r['pred']:.3f}</td><td>{r['error']:+.3f}</td><td>{abs(r['error'])/r['true']*100:.1f} %</td></tr>"
    return f"<table><tr><th>Schema</th><th>True f</th><th>ε</th><th>f(10)</th><th>Prediction</th><th>Error ŷ−f(10)</th><th>Rel. error</th></tr>{rows}</table>"


def irr_table():
    rows = ""
    for fn in FUNCS:
        for e in EPS:
            ir = R["irreducible"][f"{fn} | {e}"]
            a = R["linear"][f"(x,y) | {fn} | {e}"]["test_rmse"]
            b = R["linear"][f"(x,x^2,x^3,y) | {fn} | {e}"]["test_rmse"]
            rows += f"<tr><td>{FUNC_HTML[fn]}</td><td>{e}</td><td>{f(ir['theory'])}</td><td>{f(ir['empirical_test'])}</td><td>{f(a)}</td><td>{f(b)}</td></tr>"
    return ("<table><tr><th>True f</th><th>ε</th><th>Irreducible RMSE ε/√3</th><th>RMSE of true f on test set</th>"
            f"<th>Test RMSE, model on (x,y)</th><th>Test RMSE, model on (x,x²,x³,y)</th></tr>{rows}</table>")


def q7_table():
    q = R["q7"]
    head = "".join(f"<th>{'x' if d == 1 else 'x…x' + str(d)}</th>" for d in q["degrees"])
    tr = "".join(f"<td>{v:.5f}</td>" for v in q["train"])
    te = "".join(f"<td>{v:.5f}</td>" for v in q["test"])
    return f"<table class=small><tr><th>Columns</th>{head}</tr><tr><th>Train</th>{tr}</tr><tr><th>Test</th>{te}</tr></table>"


def tree_table():
    rows = ""
    for fn in FUNCS:
        for e in EPS:
            for s in SCHEMAS:
                for m in ["1000", "50", "1"]:
                    r = R["trees"][f"{s} | {fn} | {e} | {m}"]
                    rows += (f"<tr><td>{FUNC_HTML[fn]}</td><td>{e}</td><td>{SCH_HTML[s]}</td><td>{m}</td><td>{r['n_nodes']}</td>"
                             f"<td>{r['n_leaves']}</td><td>{r['depth']}</td><td>{r['leaf_min']}–{r['leaf_max']} (mean {r['leaf_mean']:.1f})</td>"
                             f"<td>{f(r['train_rmse'])}</td><td>{f(r['test_rmse'])}</td></tr>")
    return ("<table class=small><tr><th>True f</th><th>ε</th><th>Schema</th><th>min_records</th><th>Nodes</th><th>Leaves</th>"
            f"<th>Depth</th><th>Records per leaf</th><th>Train RMSE</th><th>Test RMSE</th></tr>{rows}</table>")


def q10_table():
    rows = ""
    for r in R["q10"]:
        vals = {"linear": r["linear"], "1000": r["tree_1000"], "50": r["tree_50"], "1": r["tree_1"]}
        best = min(vals, key=vals.get)
        cell = lambda k: f"<td{' class=win' if k == best else ''}>{f(vals[k])}</td>"
        rows += (f"<tr><td>{FUNC_HTML[r['func']]}</td><td>{r['eps']}</td><td>{SCH_HTML[r['schema']]}</td>"
                 f"{cell('linear')}{cell('1000')}{cell('50')}{cell('1')}<td>{'linear' if best == 'linear' else 'tree (m=' + best + ')'}</td></tr>")
    return ("<table><tr><th>True f</th><th>ε</th><th>Schema</th><th>Linear</th><th>Tree m=1000</th><th>Tree m=50</th>"
            f"<th>Tree m=1</th><th>Winner</th></tr>{rows}</table>")


def q11_table():
    rows = "".join(f"<tr><td>{r['min_records']}</td><td>{r['n_leaves']}</td><td>{f(r['train_rmse'],4)}</td><td>{f(r['test_rmse'],4)}</td></tr>" for r in R["q11"])
    return f"<table class=small><tr><th>min_records</th><th>Leaves</th><th>Train RMSE</th><th>Test RMSE</th></tr>{rows}</table>"


def img(name, cap):
    return f'<figure><img src="../figures/{name}"><figcaption>{cap}</figcaption></figure>'


html = f"""<!doctype html><html><head><meta charset="utf-8"><title>HW2 Report</title>
<style>
@page {{ size: A4; margin: 16mm 15mm 18mm; @bottom-center {{ content: "Page " counter(page) " of " counter(pages); font-family: "Segoe UI", Arial, sans-serif; font-size: 9pt; color: #555; }} }}
.title {{ text-align: center; margin-bottom: 14px; }}
.title .course {{ font-size: 18pt; font-weight: bold; }}
.title .sub {{ font-size: 14pt; margin-top: 2px; }}
.title .name {{ font-size: 11pt; margin-top: 8px; }}
body {{ font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.42; color: #111; }}
h1 {{ font-size: 18pt; margin: 0 0 4px; }}
h2 {{ font-size: 13pt; border-bottom: 1.5px solid #333; padding-bottom: 2px; margin: 18px 0 8px; }}
h3 {{ font-size: 11pt; margin: 12px 0 4px; }}
.q {{ break-before: page; }}
table {{ border-collapse: collapse; margin: 6px 0 10px; font-size: 9.5pt; }}
th, td {{ border: 1px solid #999; padding: 2px 6px; text-align: center; }}
th {{ background: #eee; }}
td.m {{ font-family: Consolas, monospace; font-size: 8.8pt; text-align: left; }}
td.win {{ font-weight: bold; background: #e3f1e3; }}
table.small {{ font-size: 8.4pt; }}
pre {{ background: #f6f6f6; border: 1px solid #ddd; padding: 8px 10px; font-size: 8.8pt; line-height: 1.32; white-space: pre-wrap; }}
figure {{ margin: 6px 0 10px; text-align: center; break-inside: avoid; }}
figure img {{ max-width: 100%; max-height: 225mm; }}
figcaption {{ font-size: 9pt; color: #444; }}
.note {{ background: #fff8e1; border-left: 4px solid #e0a800; padding: 6px 10px; font-size: 9.5pt; }}
ul, ol {{ margin: 4px 0 6px; padding-left: 22px; }}
</style></head><body>

<div class="title"><div class="course">CS 6220: Homework 2</div><div class="sub">Linear Regression and Regression Trees</div><div class="name">Name: Nikhil Singh Shekhawat</div></div>

<h2>Q1. Source-code repository</h2>
<p><b>GitHub:</b> <a href="{REPO_URL}">{REPO_URL}</a></p>
<p>The repository contains:</p>
<ul>
<li><code>HW2.ipynb</code>: all code, already executed so every output is visible. It is organized in markdown-headed parts: setup, data generator, linear regression, regression tree, linear-regression experiments, tree experiments, and the results dump.</li>
<li><code>README.md</code>: build and run instructions.</li>
<li><code>requirements.txt</code>: dependencies.</li>
<li><code>figures/</code>: all plots.</li>
<li><code>results/results.json</code>: all numbers quoted below.</li>
</ul>
<p><b>Data setup used throughout.</b> <code>generate_noisy_dataset</code> samples x ~ U[0,1], builds the columns (x, f<sub>1</sub>(x), …, f<sub>d</sub>(x)), and sets y = f(x) + U[−ε, ε].</p>
<ul>
<li>A fixed seed (42) is used, and x is drawn before the noise. As a result the schemas (x,y) and (x,x²,x³,y) contain <i>identical</i> x and y values.</li>
<li>Every dataset is split once with a fixed permutation (seed 7) into 800 training and 200 test records. Every linear model and every tree for a given (f, ε) therefore uses exactly the same training and test records.</li>
<li>Because the seed is shared, the linear and cubic datasets also share the same x values and noise draws. This is why some RMSEs below agree to all digits.</li>
</ul>

<h2>Q2. Pseudo-code: linear regression</h2>
<pre>
function GENERATE_NOISY_DATASET(feature_funcs, f, [a,b], n, ε, seed):
    x ← n samples from Uniform(a, b)
    X ← columns [x, f_1(x), …, f_d(x)]                 # d = |feature_funcs|, may be 0
    y ← f(x) + n samples from Uniform(−ε, ε)
    return X, y

function TRAIN_TEST_SPLIT(X, y, seed):                 # fixed partition
    π ← random permutation of 0..n−1 (seeded)
    return X[π[0:0.8n]], X[π[0.8n:]], y[π[0:0.8n]], y[π[0.8n:]]

function FIT(X_train, y_train):
    X ← [1 | X_train]                                  # step 1: prepend column of 1s; Y ← y_train
    β ← (Xᵀ X)⁻¹ Xᵀ Y                                  # step 2: normal equation (numpy)
    return β                                           # step 3: weight vector (β0 = intercept)

function PREDICT(β, X_test):
    X ← [1 | X_test]
    return X β                                         # ŷ_i = β · x_i for every record

function EVALUATE_MODEL(ŷ, y):
    return sqrt( Σ_i (y_i − ŷ_i)² / n )                # RMSE
</pre>

<h2 class="q">Q3. Linear regression on (x, y)</h2>
{img("q3_lr_xy.png", "Figure 1. Schema (x, y): training data (800 points), true function (black), and learned linear model (red, dashed).")}
{lin_table("(x,y)")}
<p>The straight line matches the linear truth almost perfectly. For the cubic truth it cannot follow the curvature: it underestimates near both ends of the range and overestimates in the middle. This bias shows up as a test RMSE of 0.0221 at ε = 0.01, about four times the noise level.</p>

<h2 class="q">Q4. Linear regression on (x, x², x³, y)</h2>
{img("q4_lr_xx2x3.png", "Figure 2. Schema (x, x², x³, y): the full learned function β₀+β₁x+β₂x²+β₃x³ is plotted.")}
{lin_table("(x,x^2,x^3,y)")}
<p><b>Do the extra columns help?</b> Yes for the cubic function, and not for the linear function.</p>
<ul>
<li><b>Cubic:</b> the model class now contains the true function. Test RMSE falls from 0.0221 to 0.0057 (ε = 0.01) and from 0.0629 to 0.0574 (ε = 0.1), which is the noise floor in both cases (see Q6c).</li>
<li><b>Linear:</b> the extra columns bring no gain. The true function already lies in the simpler model class, so the model only spends its x² and x³ coefficients fitting noise (e.g. −0.18x² + 0.14x³ at ε = 0.1). Test RMSE is unchanged or marginally worse (0.05739 → 0.05744).</li>
</ul>
<p>The deviations of the cubic-schema coefficients from the truth are identical for both true functions (+0.0587, −0.1793, +0.1350 at ε = 0.1). The reason is that f<sub>cubic</sub> − f<sub>linear</sub> lies in the span of the columns and the two datasets share the same noise draws, so least squares fits exactly the same noise in both cases.</p>

<h2 class="q">Q5. Prediction error at x = 10</h2>
{extrap_table()}
<p>The prediction is only <b>good</b> for the <b>linear true function with the (x, y) schema</b>, with errors of 0.001 and 0.008. There the model is the correct form and its two coefficients are estimated almost exactly, so extrapolation stays accurate.</p>
<ul>
<li><b>Cubic truth, (x, y) schema:</b> the straight line cannot represent the x³ growth and predicts about 4.6 instead of 410.</li>
<li><b>(x, x², x³) schema, both truths:</b> the small coefficient errors from fitting noise on [0, 1] are multiplied by x² = 100 and x³ = 1000. The prediction is off by 11.8 (ε = 0.01) or 117.7 (ε = 0.1).</li>
</ul>
<p>The ε = 0.01 cubic model is still within 3 %, but in general extrapolating far outside the training range is unreliable, especially for higher-degree models.</p>

<h2 class="q">Q6. Analysis across all linear-regression experiments</h2>
<h3>(a) For which true function does the model predict well?</h3>
<p>For the <b>linear</b> function the model predicts well with either schema, with test RMSE at the noise floor. For the <b>cubic</b> function it predicts well only with the extra columns x², x³. With (x, y) alone it has clear bias: the test RMSE is 3.9× the noise floor at ε = 0.01, and still 9 % above it at ε = 0.1, where the noise hides much of the bias.</p>
<h3>(b) Optimal model</h3>
<p>Under squared loss the optimal predictor is the conditional mean E[y | x]. Here y = f(x) + noise, and the noise is U[−ε, ε] with mean 0 and independent of x, so E[y | x] = f(x). <b>The optimal model is the true function itself:</b></p>
<ul>
<li>Linear data: f*(x) = 1 + 2x.</li>
<li>Cubic data: f*(x) = x − x² + 0.5x³.</li>
</ul>
<p>It has zero bias and zero variance and leaves only the noise.</p>
<p><b>Learned models compared with the optimum:</b></p>
<ul>
<li><b>Linear truth, (x, y):</b> 1.0002 + 2.0001x (ε = 0.01) and 1.0024 + 2.0006x (ε = 0.1), essentially the optimal model.</li>
<li><b>Cubic truth, (x, x², x³):</b> −0.0001 + 1.0059x − 1.0179x² + 0.5135x³ at ε = 0.01 is very close. At ε = 0.1 (−0.0006 + 1.0587x − 1.1793x² + 0.6350x³) the coefficients are noticeably off, because x, x² and x³ are highly correlated on [0, 1]. The resulting curve is still almost indistinguishable from f on [0, 1] (Figure 2).</li>
<li><b>Cubic truth, (x, y):</b> cannot reach the optimum, because no straight line equals the cubic.</li>
</ul>
<h3>(c) Irreducible error</h3>
<p>The irreducible error is the error of the optimal model, i.e. the noise itself: E[(y − f(x))²] = Var(U[−ε, ε]) = (2ε)²/12 = ε²/3. As an RMSE this is ε/√3. We also measured it empirically as the RMSE of the true f on each test set.</p>
{irr_table()}
<p>Every model whose hypothesis class contains the true function reaches the irreducible error on the test data, within 0.5 %:</p>
<ul>
<li>the linear truth with either schema;</li>
<li>the cubic truth with (x, x², x³).</li>
</ul>
<p>The cubic truth with (x, y) stays above the floor, with 0.0221 vs 0.0058 and 0.0629 vs 0.0577, because of its bias. No model can go below the floor on average. The small differences (e.g. 0.00574 vs 0.00577) are sampling fluctuation of the 200 test records.</p>

<h2 class="q">Q7. Adding higher-order monomials (cubic, ε = 0.1, n = 1000)</h2>
{img("q7_monomials.png", "Figure 3. Train and test RMSE as columns x², …, x¹⁰ are added; the dotted line is the irreducible error 0.1/√3.")}
{q7_table()}
<p><b>Training error</b> never increases as columns are added. Each schema contains the previous one, so least squares can always reproduce the previous fit and can only fit the training data better.</p>
<p><b>Test error</b> behaves differently:</p>
<ul>
<li>It drops sharply up to x³, because the bias disappears once the true cubic can be represented.</li>
<li>After that it barely changes (≈ 0.0570–0.0574). The extra columns mostly fit noise, so the small additional training-error reductions do not carry over to the test data.</li>
</ul>
<p>With 800 training points the variance added by a few extra coefficients is small, so clear overfitting (test error rising sharply) is not visible yet. With fewer records or higher degrees the gap between train and test error would grow. XᵀX also becomes badly conditioned (cond ≈ 5·10¹⁴ at degree 10), which makes the inverse numerically fragile.</p>

<h2 class="q">Q8. Pseudo-code: regression tree</h2>
<pre>
class NODE: n_records, value (= mean y), feature, threshold, left, right   # leaf ⇔ left = None

function FIT(X, y, min_records):
    root ← BUILD(X, y)

function BUILD(X, y):
    node ← NODE(n_records = |y|, value = mean(y))
    if |y| &lt; min_records or |y| &lt; 2:  return node            # stopping rule → leaf
    split ← BEST_SPLIT(X, y)
    if split = None:  return node                              # all y equal / all x equal → leaf
    (j, t) ← split
    L ← {{i : X[i, j] ≤ t}};   R ← {{i : X[i, j] &gt; t}}
    node.feature, node.threshold ← j, t
    node.left  ← BUILD(X[L], y[L])                             # recurse
    node.right ← BUILD(X[R], y[R])
    return node

function BEST_SPLIT(X, y):
    best ← (∞, None)
    for each column j:
        sort records by X[:, j]  → x_s, y_s
        prefix sums S1[k] = Σ_{{i≤k}} y_s[i],  S2[k] = Σ_{{i≤k}} y_s[i]²
        for k = 1 .. n−1 with x_s[k] &lt; x_s[k+1]:              # candidate cut between distinct values
            SSE_L = S2[k] − S1[k]²/k
            SSE_R = (S2[n]−S2[k]) − (S1[n]−S1[k])²/(n−k)
            if SSE_L + SSE_R &lt; best.sse:
                best ← (SSE_L + SSE_R, (j, (x_s[k] + x_s[k+1]) / 2))
    return best.split

function PREDICT(X):
    for each record x:  node ← root
        while node is not a leaf:  node ← node.left if x[node.feature] ≤ node.threshold else node.right
        output node.value

function VISUALIZE(ax, x_range, feature_funcs):
    grid ← dense x values in x_range;  G ← [grid, f_1(grid), …]
    draw step plot of PREDICT(G) against grid
    for each internal node (j, t):  x_split ← t if j = 0 else the x with f_j(x) = t   # monotone on [0,1]
        draw vertical dashed line at x_split
</pre>

<h2 class="q">Q9. Regression trees: all setup combinations</h2>
<p>The trees use the same 800/200 splits as the linear models. In each figure the rows are the schemas and the columns are min_records = 1000, 50, 1. Each panel shows the training data, the true function (black), the tree prediction (red step line), and the split points (dashed gray). At min_records = 1 there are 799 split lines, so they form a gray band.</p>
{img("q9_tree_linear_eps0.01.png", "Figure 4. Linear f, ε = 0.01.")}
{img("q9_tree_linear_eps0.1.png", "Figure 5. Linear f, ε = 0.1.")}
{img("q9_tree_cubic_eps0.01.png", "Figure 6. Cubic f, ε = 0.01.")}
{img("q9_tree_cubic_eps0.1.png", "Figure 7. Cubic f, ε = 0.1.")}
<h3>Tree statistics and errors</h3>
{tree_table()}

<h2 class="q">Q10. Trees vs. linear models (test RMSE)</h2>
{q10_table()}
<p><b>Who wins?</b> The <b>linear model wins whenever its schema contains the true function</b>: the linear truth with both schemas and the cubic truth with (x, x², x³). It then sits at the noise floor, which a piecewise-constant tree cannot reach with 800 records. The <b>tree wins only for the cubic truth on (x, y)</b>, where the straight line is biased: 0.0079 vs 0.0221 at ε = 0.01, and 0.062 vs 0.063 at ε = 0.1.</p>
<p><b>(i) Extra columns.</b> They make <b>no difference</b> to the trees: every tree, RMSE, node count, and leaf is identical for the two schemas.</p>
<ul>
<li>On [0, 1], x² and x³ are strictly increasing functions of x.</li>
<li>So a threshold on x² or x³ produces exactly the same left/right partition as some threshold on x.</li>
<li>Trees only use the ordering of a column, so they gain nothing; our implementation always chose column x.</li>
</ul>
<p>Linear regression, by contrast, benefits from the extra columns because it uses their actual values.</p>
<p><b>(ii) Tree size.</b> Size matters a lot:</p>
<ul>
<li><b>min_records = 1000:</b> the 800-record root is never split, giving <b>1 node</b> (a single leaf with 800 records). The prediction is a constant (the mean of y), with high bias: test RMSE 0.60 (linear) and 0.14 (cubic).</li>
<li><b>min_records = 50:</b> <b>47–73 nodes</b> (24–37 leaves, depth 5–9). Leaves hold mostly 12–49 records, occasionally fewer, because a split of a ≥ 50-record node can produce a very small child. This gives a good staircase approximation.</li>
<li><b>min_records = 1:</b> splitting continues until every leaf is pure, giving <b>1599 nodes</b> (800 leaves of exactly 1 record each, depth 15–22). Training RMSE is 0 because the tree memorizes the data, noise included.</li>
</ul>
<p>The best size is in between and depends on the noise.</p>
<p><b>(iii) Noise level.</b> The noise level decides which tree size is best:</p>
<ul>
<li><b>ε = 0.01:</b> memorizing the noise is cheap, and the main error comes from the step approximation. The fully grown tree (m = 1) is therefore the best tree (0.0080 linear, 0.0079 cubic), clearly better than m = 50 (0.025, 0.0087).</li>
<li><b>ε = 0.1:</b> the noise dominates. A 1-record leaf passes its noise directly into predictions, giving test RMSE 0.079 ≈ √2·ε/√3. The m = 50 tree averages the noise over many records per leaf and wins (0.064, 0.062).</li>
</ul>
<p>Noise also raises every model's error floor (×10 from ε = 0.01 to 0.1), which narrows the gap between the trees and the biased linear model.</p>

<h2 class="q">Q11. Overfitting study: f(x) = sin(2πx), n = 200, ε = 0.2</h2>
{img("q11_overfitting.png", "Figure 8. Train and test RMSE vs. min_records (160 training / 40 test records). Dotted: irreducible error 0.2/√3 ≈ 0.115.")}
{q11_table()}
{img("q11_trees.png", "Figure 9. The fitted trees for each min_records value.")}
<ul>
<li><b>Best model: min_records = 8</b>, with the lowest test RMSE (0.136, 37 leaves). It follows the sine wave without chasing individual noisy points. Values 16–32 are close behind (0.161–0.164); with only 40 test records, these differences are partly noise.</li>
<li><b>Highest bias: min_records = 256.</b> With only 160 training records the root is never split, so the tree is the constant mean of y. It cannot represent the sine shape at all, and both errors are high (train 0.739, test 0.750, nearly equal). min_records = 128 (2 leaves) is also strongly biased.</li>
<li><b>Highest variance: min_records = 2.</b> The tree grows until every leaf holds one record (160 leaves), so it fits the noise exactly: training RMSE is 0 but test RMSE is 0.175. A different noise sample would produce a completely different tree.</li>
</ul>
<p>In between, training error rises steadily with min_records, while test error first falls and then rises: the classic U-shaped bias-variance trade-off.</p>

<h2>Q12. Hyper-parameters</h2>
<p><b>Linear regression:</b> the training algorithm itself has <b>no tunable hyper-parameters</b>. β comes from the closed-form normal equation, so there is no learning rate, number of iterations, or regularization strength. The only model-complexity knob is the <b>choice of input columns</b> (the <code>feature_funcs</code>, e.g. the highest monomial degree in Q7). It is a modelling or feature-engineering choice that behaves like a hyper-parameter: it controls the bias-variance trade-off.</p>
<p><b>Regression tree:</b> one hyper-parameter, <b><code>min_records</code></b>. A node is split only if it holds at least that many training records, so it controls tree size and hence bias vs. variance. Everything else is fixed by design: SSE split criterion, thresholds at midpoints, no maximum depth, and no pruning.</p>
<p><b>Not hyper-parameters:</b> the data-generation settings (range, sample count, noise ε), the 80/20 split fraction, and the random seeds are experiment settings, not hyper-parameters of either algorithm.</p>
</body></html>"""

out_html = os.path.join(ROOT, "report", "report.html")
with open(out_html, "w", encoding="utf-8") as fh:
    fh.write(html)
print("wrote", out_html)

edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
pdf = os.path.join(ROOT, "HW2_Report.pdf")
subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}", "file:///" + out_html.replace("\\", "/")], check=True)
print("wrote", pdf)
