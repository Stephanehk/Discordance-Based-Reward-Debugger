import json, html, re, pathlib
ROOT = pathlib.Path('/next/u/stephhk/automated_reward_repairer')
RUN = pathlib.Path('/next/u/stephhk/traj_pair_comparison_automated_reward_repairer/user_study_rfs/metaworld_drawer_open/our_system_collected_data/default_rf')
SNAP = json.loads((RUN/'shown_to_user/pref_round_001.json').read_text())
SNAP_L = json.loads((RUN/'shown_to_user/pref_round_003.json').read_text())
FEATURE_KEY = json.loads((RUN/'shown_to_user/_ui_payload/feature_key.json').read_text())
OUT = ROOT/'docs/walkthrough.html'

def feats(traj_path):
    k = traj_path.split('/')[1].split('_')[1]
    d = json.loads((RUN/f'rollouts/round_{k}_trajectory_features'/(pathlib.Path(traj_path).stem+'.json')).read_text())
    return dict(zip([n[0] for n in d['names']], d['features']))

def fmt(v):
    return str(int(round(v))) if abs(v-round(v))<1e-9 else f'{v:.2f}'

def bullets(fa, fb):
    out=[]
    for f in FEATURE_KEY:
        a,b = fa[f['name']], fb[f['name']]
        word = 'higher' if a>b+1e-9 else 'lower' if a<b-1e-9 else 'same'
        out.append(f'**{word}** {f["description"]} ({fmt(a)} vs. {fmt(b)})')
    return out

def esc(s): return html.escape(s, quote=False)
def inline(s):
    s = esc(s)
    return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)

NPAIRS = len(SNAP['pairs']); NLOTS = len(SNAP_L['lotteries']); NALL = NPAIRS + NLOTS
def pair_card(n, open_):
    p = SNAP['pairs'][n-1]
    head = "Designed reward function's preferred trajectory has a"
    bullets_ = bullets(feats(p['traj_a_path']), feats(p['traj_b_path']))
    rets = {'A': p['snapshot_reward_a'], 'B': p['snapshot_reward_b']}
    vids = ''.join(f'''
          <figure>
            <video controls muted loop playsinline preload="metadata" src="walkthrough/media/pair{n}-{r}.mp4"></video>
            <figcaption>{cap}</figcaption>
            <div class="viz-return">Return: {rets[r]:.2f}</div>
          </figure>''' for r, cap in (('A',"Designed reward function's preferred trajectory"),('B','The alternative trajectory')))
    lis = ''.join(f'\n          <li>{inline(b)}</li>' for b in bullets_)
    return f'''
      <article class="msg msg-cmp">
        <details class="cmp-details"{" open" if open_ else ""}>
          <summary class="msg-head"><span class="label">DISCORDANCE {n}/{NALL}</span></summary>
          <div class="cmp-viz">{vids}
          </div>
          <details class="cmp-more">
            <summary>How the two trajectories differ in more detail</summary>
            <div class="cmp-chars-head">{inline(head)}</div>
            <ul class="cmp-chars">{lis}
            </ul>
          </details>
        </details>
      </article>'''

def lottery_card(n, open_):
    l = SNAP_L['lotteries'][n-1]
    pa = round(l['p']*100); pb = 100-pa
    prefC = bool(l['snapshot_reward_prefers_c'])
    # Per-trajectory returns: C's is stored in the snapshot; A's and B's are
    # quoted in the selection justification as "T### (return)".
    _quoted = dict(re.findall(r'T(\d+) \(([\d.]+)\)', l.get('justification', '')))
    _ids = [k for k in _quoted]  # order: C, A, B
    rets = {'C': float(l['snapshot_reward_c'])}
    if len(_ids) >= 3:
        rets['A'] = float(_quoted[_ids[1]]); rets['B'] = float(_quoted[_ids[2]])
    def fig(role, cap, prob=None):
        p = f'\n            <div class="viz-prob">{prob}% chance</div>' if prob is not None else ''
        r = f'\n            <div class="viz-return">Return: {rets[role]:.2f}</div>' if role in rets else ''
        return f'''
          <figure>
            <video controls muted loop playsinline preload="metadata" src="walkthrough/media/lottery{n}-{role}.mp4"></video>
            <figcaption>{cap}</figcaption>{p}{r}
          </figure>'''
    badge = '<div class="viz-preferred lot-preferred">Designed reward function assigns higher return to this certain trajectory</div>' if prefC else '<div class="viz-preferred lot-preferred">Designed reward function assigns higher return to this lottery</div>'
    ab_row = f'<div class="lot-row lot-ab">{fig("A","Trajectory A",pa)}{fig("B","Trajectory B",pb)}\n          </div>'
    c_row = f'<div class="lot-row lot-c">{fig("C","Certain (C)")}\n          </div>'
    body = (ab_row + '\n          ' + badge + '\n          ' + c_row) if prefC else (badge + '\n          ' + ab_row + '\n          ' + c_row)
    return f'''
      <article class="msg msg-cmp msg-lottery">
        <details class="cmp-details"{" open" if open_ else ""}>
          <summary class="msg-head"><span class="label">DISCORDANCE {NPAIRS+n}/{NALL} (lottery)</span></summary>
          <div class="lot-viz">
          {body}
          </div>
        </details>
      </article>'''

code = (RUN/'designed_reward_fn.py').read_text()

SETUP_FEED = '''
      <article class="msg msg-status"><div class="msg-body">The environment is set up.</div></article>
      <article class="msg msg-md"><div class="msg-body md">
        <p><strong>What the agent perceives.</strong> The agent controls a Sawyer robot arm. It senses where its own hand is in space, how far the gripper is open or closed, and where the drawer handle is (both its position and the way it is oriented). It also knows where the handle needs to end up for the drawer to count as fully open, where its hand started at the beginning of the attempt, and the overall goal location. All positions are measured in meters.</p>
        <p><strong>What the agent can do.</strong> On each step the agent nudges its hand a small amount along three directions — left/right, forward/back, and up/down — and sends a command to open or close the gripper. Repeating these small moves is how it reaches the handle and pulls the drawer open.</p>
      </div></article>
      <article class="msg msg-md"><div class="msg-body md"><p>A stub reward function is in place. Fill in its body in <code>workspaces/metaworld_drawer_open/designed_reward_fn.py</code>, then press the button below.</p></div></article>
      <article class="msg msg-cta" data-cta="{mode}">
        <div class="cta-prompt muted">The environment is set up. Press when your reward function is written.</div>
        <button type="button" class="btn btn-cta">Identify possible flaws</button>
      </article>
      <div class="thinking" hidden><span class="spinner"></span><span>Claude is thinking…</span></div>
      <div class="msg msg-status after-cta" hidden><div class="msg-body"></div></div>'''
def setup_feed(mode): return SETUP_FEED.replace('{mode}', mode)

def frame(inner, status='awaiting', url='127.0.0.1:8000'):
    return f'''
    <div class="frame">
      <div class="frame-bar"><span class="fdots"><i></i><i></i><i></i></span><span class="furl">{url}</span></div>
      <div class="app">
        <header class="topbar">
          <div class="topbar-left"><div class="dot"></div><span class="brand">Automated Reward Repairer</span></div>
          <div class="topbar-right"><span class="muted">workspace</span><span class="chip">metaworld_drawer_open</span><span class="muted">·</span><span class="chip status" data-status="{status}">{status}</span></div>
        </header>
        <div class="feed">{inner}
        </div>
      </div>
    </div>'''

QUESTION1 = '''
      <article class="msg msg-q" data-q="specify">
        <header class="msg-head"><span class="label">Question</span><span class="qstatus muted">awaiting your answer</span></header>
        <div class="q-prompt">How would you like to specify the reward function?</div>
        <div class="q-body">
          <div class="q-options">
            <button type="button" class="q-option" data-value="describe"><span class="opt-label">Describe it — I will implement</span><span class="opt-desc">Fill in a short template; I translate it into Python and show it back for review.</span></button>
            <button type="button" class="q-option" data-value="myself"><span class="opt-label">I will write it myself</span><span class="opt-desc">I show a stub with the observation/action summary as a docstring; you fill in the body here or in the file.</span></button>
          </div>
          <div class="q-actions"><button type="button" class="btn">Submit</button></div>
        </div>
        <div class="q-note" hidden></div>
      </article>'''

QUESTION2 = '''
      <article class="msg msg-q" data-q="reward">
        <header class="msg-head"><span class="label">Question</span><span class="qstatus muted">awaiting your answer</span></header>
        <div class="q-prompt">Write your reward function below (Python), or paste an existing one. You can also edit the file directly at <code>workspaces/metaworld_drawer_open/designed_reward_fn.py</code> in the repo and submit the box as is.</div>
        <div class="q-body">
          <textarea class="q-textarea q-code" spellcheck="false">&quot;&quot;&quot;Designed reward for Meta-World drawer-open-v3.

Observation (45 floats): hand position (x, y, z); gripper aperture; drawer handle
position and orientation; target handle position when fully open; hand position at
the start of the episode. All positions in meters.
Action (4 floats): delta x, y, z of the hand; gripper command (&gt; 0 closes).

Fill in the body. Signature must be unchanged.
&quot;&quot;&quot;

def designed_reward_fn(obs, act, next_obs) -&gt; float:
    raise NotImplementedError(&quot;Fill in designed_reward_fn before running /repair-init.&quot;)</textarea>
          <div class="q-actions"><button type="button" class="btn">Submit</button></div>
        </div>
        <div class="q-note" hidden></div>
      </article>'''

QUESTION7 = '''
      <article class="msg msg-q" data-q="next">
        <header class="msg-head"><span class="label">Question</span><span class="qstatus muted">awaiting your answer</span></header>
        <div class="q-prompt">What would you like to do next?</div>
        <div class="q-body">
          <div class="q-options">
            <button type="button" class="q-option" data-value="edit"><span class="opt-label">Edit the reward function</span><span class="opt-desc">Modify the reward to reflect what you actually want, based on the discordances above.</span></button>
            <button type="button" class="q-option" data-value="more"><span class="opt-label">Propose more discordances</span><span class="opt-desc">Propose more pairs or lotteries from the trajectories already collected.</span></button>
            <button type="button" class="q-option" data-value="finish"><span class="opt-label">End session</span><span class="opt-desc">Keep the current reward function and close the tool.</span></button>
          </div>
          <div class="q-actions"><button type="button" class="btn">Submit</button></div>
        </div>
        <div class="q-note" hidden></div>
      </article>'''

ROBOT_SVG = '''<svg viewBox="0 0 140 128" width="140" height="128" aria-hidden="true">
  <defs><clipPath id="beebody"><ellipse cx="70" cy="98" rx="21" ry="19"/></clipPath></defs>
  <g class="wing wing-l"><ellipse cx="30" cy="46" rx="25" ry="14" transform="rotate(-24 30 46)" fill="#fff" fill-opacity=".95" stroke="#3a2e2a" stroke-width="3"/><ellipse cx="30" cy="46" rx="13" ry="6" transform="rotate(-24 30 46)" fill="#eef4ff"/></g>
  <g class="wing wing-r"><ellipse cx="110" cy="46" rx="25" ry="14" transform="rotate(24 110 46)" fill="#fff" fill-opacity=".95" stroke="#3a2e2a" stroke-width="3"/><ellipse cx="110" cy="46" rx="13" ry="6" transform="rotate(24 110 46)" fill="#eef4ff"/></g>
  <g class="body">
    <path d="M66 114 L70 125 L74 114 Z" fill="#3a2e2a"/>
    <ellipse cx="70" cy="98" rx="21" ry="19" fill="#fbd34d"/>
    <g clip-path="url(#beebody)"><rect x="40" y="90" width="60" height="7" fill="#3a2e2a"/><rect x="40" y="103" width="60" height="7" fill="#3a2e2a"/></g>
    <ellipse cx="70" cy="98" rx="21" ry="19" fill="none" stroke="#3a2e2a" stroke-width="3"/>
    <path d="M56 26 q-6 -14 -15 -14" fill="none" stroke="#3a2e2a" stroke-width="3" stroke-linecap="round"/>
    <path d="M84 26 q6 -14 15 -14" fill="none" stroke="#3a2e2a" stroke-width="3" stroke-linecap="round"/>
    <circle class="lamp" cx="40" cy="11" r="4.5" fill="#fbd34d" stroke="#3a2e2a" stroke-width="3"/>
    <circle class="lamp" cx="100" cy="11" r="4.5" fill="#fbd34d" stroke="#3a2e2a" stroke-width="3"/>
    <circle cx="70" cy="54" r="32" fill="#fbd34d" stroke="#3a2e2a" stroke-width="3"/>
    <g class="eyes">
      <circle cx="58" cy="54" r="9" fill="#fff"/><circle cx="82" cy="54" r="9" fill="#fff"/>
      <circle cx="59.5" cy="55.5" r="5.5" fill="#2b2a33"/><circle cx="83.5" cy="55.5" r="5.5" fill="#2b2a33"/>
      <circle cx="61.5" cy="53" r="2" fill="#fff"/><circle cx="85.5" cy="53" r="2" fill="#fff"/>
    </g>
    <circle cx="46" cy="66" r="4.5" fill="#f6a9a0" fill-opacity=".85"/><circle cx="94" cy="66" r="4.5" fill="#f6a9a0" fill-opacity=".85"/>
    <path d="M64 68 q6 5 12 0" fill="none" stroke="#3a2e2a" stroke-width="2.5" stroke-linecap="round"/>
    <circle cx="54" cy="40" r="1.6" fill="#3a2e2a" opacity=".55"/><circle cx="86" cy="40" r="1.6" fill="#3a2e2a" opacity=".55"/>
  </g>
</svg>'''

def with_buddy(frame_html, text):
    return f'''
    <div class="frame-wrap">{frame_html}
      <div class="buddy" aria-live="polite">
        <div class="bubble">{text}</div>
        <div class="bot">{ROBOT_SVG}</div>
      </div>
    </div>'''

PAIR_NOTE = "Notice how the reward function assigns higher return to a trajectory where the robot doesn't open the drawer over one where it does."
_l1 = SNAP_L['lotteries'][0]; _pfail = 100 - round(_l1['p']*100)
LOT_NOTE = f"And here, the reward function assigns higher return to a lottery with a {_pfail}% chance that the robot doesn't open the drawer over a trajectory where it opens the drawer for certain. This gives you information about the reward function's cardinal ranking of trajectories."

EDITOR = f'''
    <div class="frame ide">
      <div class="frame-bar dark"><span class="fdots"><i></i><i></i><i></i></span><span class="ftab">designed_reward_fn.py</span><span class="fpath">workspaces / metaworld_drawer_open</span></div>
      <div class="editor"><pre><code class="py">{esc(code)}</code></pre></div>
    </div>'''

JS = r"""<script>
(function(){
  // ---- Python highlighter (same token classes as the app's) + line numbers
  var KW=new Set("False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield match case".split(" "));
  var BI=new Set("abs all any bool bytes callable chr complex dict dir divmod enumerate eval exec filter float format frozenset getattr globals hasattr hash help hex id input int isinstance issubclass iter len list locals map max min next object oct open ord pow print property range repr reversed round set setattr slice sorted staticmethod str sum super tuple type vars zip".split(" "));
  function esc(s){return s.replace(/[&<>"']/g,function(c){return{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]})}
  function hl(src){
    var out=[],i=0,n=src.length,m;
    function emit(t,c){out.push(c?'<span class="tok-'+c+'">'+esc(t)+'</span>':esc(t))}
    while(i<n){
      var rest=src.slice(i);
      if((m=rest.match(/^[rbuRBUfF]{0,2}("{3}|'{3})/))){var q=m[1],e=src.indexOf(q,i+m[0].length),stop=e<0?n:e+3;emit(src.slice(i,stop),"string");i=stop;continue}
      if((m=rest.match(/^[rbuRBUfF]{0,2}(["'])/))){var q2=m[1],j=i+m[0].length;while(j<n){if(src[j]==="\\"){j+=2;continue}if(src[j]===q2||src[j]==="\n")break;j++}if(j<n&&src[j]===q2)j++;emit(src.slice(i,j),"string");i=j;continue}
      if(src[i]==="#"){var eol=src.indexOf("\n",i),s2=eol<0?n:eol;emit(src.slice(i,s2),"comment");i=s2;continue}
      if((m=rest.match(/^(?:0[xX][0-9a-fA-F_]+|\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][-+]?\d+)?)/))){emit(m[0],"number");i+=m[0].length;continue}
      if((m=rest.match(/^@[A-Za-z_][\w.]*/))){emit(m[0],"decorator");i+=m[0].length;continue}
      if((m=rest.match(/^[A-Za-z_]\w*/))){var w=m[0],after=src.slice(i+w.length).match(/^\s*\(/);var prev=src.slice(0,i).match(/(def|class)\s+$/);
        emit(w,KW.has(w)?"keyword":BI.has(w)?"builtin":(prev||after)?"function":null);i+=w.length;continue}
      emit(src[i],null);i++;
    }
    return out.join("");
  }
  document.querySelectorAll(".editor code.py").forEach(function(c){
    var lines=hl(c.textContent).split("\n");
    c.innerHTML=lines.map(function(l,k){return '<span class="ln">'+(k+1)+'</span>'+l}).join("\n");
  });
  // Start the editor scrolled to the bottom, where designed_reward_fn lives.
  document.querySelectorAll(".editor").forEach(function(e){e.scrollTop=e.scrollHeight});

  // ---- Question cards: select an option, submit, show what happens next
  var NOTES={
    describe:'With this choice you fill in three short text boxes, and Claude will implement a reward function in Python for you. This tour follows the other path. <a href="#s2">Continue to step 2 &darr;</a>',
    edit:'Claude asks whether you want to edit the file yourself or describe the change in plain language. Either way, the comparisons are re-scored against the new reward.',
    more:'More discordances appear from the trajectories already collected. No new rollouts are needed.',
    finish:'The session ends and the current reward function is kept.'
  };
  document.querySelectorAll(".app .msg-q").forEach(function(card){
    var sel=null,opts=card.querySelectorAll(".q-option"),ta=card.querySelector(".q-textarea"),submit=card.querySelector(".q-actions .btn"),note=card.querySelector(".q-note"),st=card.querySelector(".qstatus");
    opts.forEach(function(o){o.addEventListener("click",function(){if(o.disabled)return;if(sel)sel.classList.remove("selected");o.classList.add("selected");sel=o})});
    submit.addEventListener("click",function(){
      if(!sel&&!ta)return;
      opts.forEach(function(o){o.disabled=true});submit.disabled=true;if(ta)ta.disabled=true;
      st.textContent="answered";st.classList.add("answered");
      note.innerHTML=NOTES[sel?sel.dataset.value:card.dataset.q]||"";note.hidden=false;
      var chip=card.closest(".app").querySelector(".chip.status");if(chip){chip.textContent="running";chip.dataset.status="running"}
    });
  });

  // ---- CTA: pressing "Identify possible flaws" starts the pipeline
  document.querySelectorAll(".app [data-cta] .btn-cta").forEach(function(btn){
    btn.addEventListener("click",function(){
      var card=btn.closest("[data-cta]"),app=btn.closest(".app"),th=app.querySelector(".thinking"),after=app.querySelector(".after-cta"),chip=app.querySelector(".chip.status");
      if(card.dataset.cta==="early"){
        after.querySelector(".msg-body").innerHTML='Not yet! You need to implement the reward function first. <a href="#s3">Scroll down to step 3 &darr;</a>';
        after.hidden=false;
        return;
      }
      btn.disabled=true;
      if(chip){chip.textContent="running";chip.dataset.status="running"}
      th.hidden=false;
      setTimeout(function(){after.querySelector(".msg-body").innerHTML='Subagents are now searching for flaws in your reward function. <a href="#s5">Move on to step 5 &darr;</a>';after.hidden=false},900);
    });
  });

  // ---- Robot buddy: fly in when the step's first video scrolls into view
  document.querySelectorAll(".buddy").forEach(function(bd){
    var wrap=bd.closest(".frame-wrap"), vid=wrap&&wrap.querySelector("video");
    if(!vid){bd.classList.add("on");return}
    if(!("IntersectionObserver" in window)){bd.classList.add("on");return}
    var o=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting){bd.classList.add("on");o.disconnect()}})},{threshold:.3});
    o.observe(vid);
  });

  // ---- Videos: play while visible, pause when not
  if("IntersectionObserver" in window){
    var io=new IntersectionObserver(function(es){es.forEach(function(e){var v=e.target;if(e.isIntersecting){v.play().catch(function(){})}else{v.pause()}})},{threshold:.35});
    document.querySelectorAll(".app video").forEach(function(v){io.observe(v)});
    document.querySelectorAll(".app details").forEach(function(d){d.addEventListener("toggle",function(){d.querySelectorAll("video").forEach(function(v){d.open?v.play().catch(function(){}):v.pause()})})});
  }
})();
</script>"""

page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Walkthrough — Discordance-Based Reward Debugger</title>
<meta name="description" content="A short tour of what you will see when you run the Discordance-Based Reward Debugger, with the real interface components embedded.">
<style>
  :root{{
    --bg:#fbfaf7; --fg:#1c1b19; --muted:#66625b; --rule:#e4e0d8;
    --accent:#1f6f5f; --accent-bg:#eef5f2; --accent-fg:#ffffff;
    --code-bg:#f1efe9; --code-fg:#1c1b19;
    --shot-shadow:0 12px 32px rgba(28,27,25,.12), 0 1px 2px rgba(28,27,25,.08);
    --frame-chrome:#ecebe6; --frame-chrome-fg:#6b675f;
    --serif:"Iowan Old Style","Palatino Linotype",Palatino,Charter,Georgia,"Times New Roman",serif;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
  }}
  @media (prefers-color-scheme: dark){{
    :root:not([data-theme="light"]){{
      --bg:#17181a; --fg:#ebe8e1; --muted:#a39e94; --rule:#2e2f33;
      --accent:#7fc7b3; --accent-bg:#1d2b27; --accent-fg:#101513;
      --code-bg:#1f2023; --code-fg:#ebe8e1;
      --shot-shadow:0 12px 32px rgba(0,0,0,.45), 0 1px 2px rgba(0,0,0,.4);
      --frame-chrome:#26272b; --frame-chrome-fg:#a39e94;
      color-scheme:dark;
    }}
  }}
  :root[data-theme="dark"]{{
    --bg:#17181a; --fg:#ebe8e1; --muted:#a39e94; --rule:#2e2f33;
    --accent:#7fc7b3; --accent-bg:#1d2b27; --accent-fg:#101513;
    --code-bg:#1f2023; --code-fg:#ebe8e1;
    --shot-shadow:0 12px 32px rgba(0,0,0,.45), 0 1px 2px rgba(0,0,0,.4);
    --frame-chrome:#26272b; --frame-chrome-fg:#a39e94;
    color-scheme:dark;
  }}

  *{{box-sizing:border-box}}
  html{{scroll-behavior:smooth}}
  body{{margin:0;background:var(--bg);color:var(--fg);font:17px/1.6 var(--sans);-webkit-font-smoothing:antialiased}}
  a{{color:var(--accent);text-decoration:none;border-bottom:1px solid transparent}}
  a:hover{{border-bottom-color:currentColor}}
  code{{font-family:var(--mono);font-size:.9em;background:var(--code-bg);padding:.1em .35em;border-radius:4px}}
  h1,h2,h3{{font-family:var(--serif);font-weight:500;letter-spacing:-.01em;margin:0}}
  h1{{font-size:clamp(34px,5.4vw,46px);line-height:1.12}}
  h2{{font-size:28px;line-height:1.2;margin-bottom:22px}}
  h3{{font-size:21px;line-height:1.3;margin-bottom:8px}}
  p{{margin:0 0 14px}}
  p:last-child{{margin-bottom:0}}
  .muted{{color:var(--muted)}}
  b{{font-weight:600}}

  .page{{max-width:1040px;margin:0 auto;padding:0 20px}}
  .text{{max-width:660px}}

  header.site{{padding:calc(56px + env(safe-area-inset-top,0px)) 0 44px;border-bottom:1px solid var(--rule)}}
  header.site nav{{display:flex;gap:22px;flex-wrap:wrap;margin-top:24px;font-size:15.5px}}
  header.site nav a{{font-weight:600}}
  .crumb{{font-size:14px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin-bottom:14px}}
  .crumb a{{color:var(--muted)}}
  .draft{{display:inline-block;margin-left:8px;padding:1px 8px;border-radius:999px;background:var(--accent-bg);color:var(--accent);font-size:11px;letter-spacing:.06em;font-weight:700;vertical-align:middle}}

  section{{padding:48px 0;border-bottom:1px solid var(--rule)}}
  section:last-of-type{{border-bottom:0}}

  ol.tour{{list-style:none;margin:0;padding:0;counter-reset:step}}
  ol.tour>li{{counter-increment:step;display:grid;grid-template-columns:72px 1fr;gap:0 18px;padding:36px 0;border-top:1px solid var(--rule)}}
  ol.tour>li:first-child{{border-top:0;padding-top:0}}
  ol.tour>li::before{{content:counter(step);font-family:var(--serif);font-size:40px;line-height:1;color:var(--accent);padding-top:2px}}
  ol.tour>li>div{{min-width:0}}
  ol.tour .note{{max-width:660px}}
  @media (max-width:640px){{
    ol.tour>li{{grid-template-columns:1fr;gap:6px}}
    ol.tour>li::before{{font-size:26px}}
  }}
  .try{{display:inline-block;margin-top:6px;font-size:13.5px;color:var(--muted)}}
  .try::before{{content:"✦ ";color:var(--accent)}}

  dl{{margin:0}} dl dt{{font-weight:600;margin-top:14px}} dl dd{{margin:2px 0 0;color:var(--muted)}}
  .aside{{margin-top:20px;padding:16px 18px;border-left:3px solid var(--accent);background:var(--accent-bg);border-radius:0 8px 8px 0;font-size:16px}}
  .callout{{margin-top:16px;padding:14px 16px;border:1px solid var(--rule);border-radius:8px;font-size:15.5px}}

  .cta{{display:inline-flex;align-items:center;gap:10px;margin-top:22px;padding:14px 24px;border-radius:10px;
       background:var(--accent);color:var(--accent-fg);font-weight:600;font-size:17px;border-bottom:0;
       box-shadow:0 6px 18px rgba(31,111,95,.22);transition:transform .12s ease,box-shadow .12s ease}}
  .cta:hover{{border-bottom:0;transform:translateY(-1px);box-shadow:0 10px 24px rgba(31,111,95,.3)}}

  footer{{padding:36px 0 calc(48px + env(safe-area-inset-bottom,0px));color:var(--muted);font-size:15px;border-top:1px solid var(--rule)}}
  footer .links{{display:flex;gap:22px;flex-wrap:wrap}}

  /* ===== Flying robot buddy (sits beside the figure, stays level while you scroll) ===== */
  .frame-wrap{{display:grid;grid-template-columns:minmax(0,1fr) 190px;gap:16px;align-items:start;overflow-x:clip}}
  .frame-wrap .frame{{min-width:0}}
  .buddy{{position:sticky;top:calc(28vh + env(safe-area-inset-top,0px));display:flex;flex-direction:column;align-items:center;gap:14px;margin-top:20px;padding:6px 0 10px}}
  .buddy .bubble{{position:relative;width:100%;padding:12px 14px;border-radius:14px;background:#fff;color:#1c1b19;border:2px solid #2f3340;font-size:14.5px;line-height:1.45;
     opacity:0;transform:scale(.7) translateY(8px);transform-origin:50% 100%}}
  .buddy .bubble::before,.buddy .bubble::after{{content:"";position:absolute;left:50%;margin-left:-8px;bottom:-10px;border:8px solid transparent;border-top-color:#2f3340;border-bottom:0}}
  .buddy .bubble::after{{bottom:-7px;border-top-color:#fff}}
  .buddy .bot{{flex:none;opacity:0;transform:translateX(260px) translateY(-24px) rotate(10deg)}}
  .buddy .bot svg{{display:block;overflow:visible;filter:drop-shadow(0 8px 12px rgba(58,46,42,.18))}}
  .buddy .wing{{transform-box:view-box}}
  .buddy .wing-l{{transform-origin:52px 48px}} .buddy .wing-r{{transform-origin:88px 48px}}
  .buddy .eyes{{transform-box:fill-box;transform-origin:50% 50%}}
  .buddy.on .bot{{animation:flyin 1.3s cubic-bezier(.22,.9,.3,1) forwards}}
  .buddy.on .bot svg{{animation:hover 2.6s ease-in-out 1.3s infinite}}
  .buddy.on .wing-l{{animation:flapL .13s ease-in-out infinite alternate}}
  .buddy.on .wing-r{{animation:flapR .13s ease-in-out infinite alternate}}
  .buddy.on .eyes{{animation:blink 4.5s ease-in-out 2s infinite}}

  .buddy.on .bubble{{animation:pop .5s cubic-bezier(.3,1.4,.5,1) 1.15s forwards}}
  @keyframes flyin{{0%{{opacity:0;transform:translateX(260px) translateY(-24px) rotate(10deg)}}15%{{opacity:1}}60%{{transform:translateX(-10px) translateY(6px) rotate(-5deg)}}100%{{opacity:1;transform:translateX(0) translateY(0) rotate(0)}}}}
  @keyframes hover{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-6px)}}}}
  @keyframes flapL{{from{{transform:rotate(16deg)}}to{{transform:rotate(-18deg)}}}}
  @keyframes flapR{{from{{transform:rotate(-16deg)}}to{{transform:rotate(18deg)}}}}
  @keyframes blink{{0%,92%,100%{{transform:scaleY(1)}}95%{{transform:scaleY(.1)}}}}
  @keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.55}}}}

  @keyframes pop{{from{{opacity:0;transform:scale(.7) translateY(8px)}}to{{opacity:1;transform:scale(1) translateY(0)}}}}
  @media (prefers-color-scheme: dark){{:root:not([data-theme="light"]) .buddy .bubble{{background:#23242a;color:#ebe8e1;border-color:#a39e94}} :root:not([data-theme="light"]) .buddy .bubble::after{{border-top-color:#23242a}} :root:not([data-theme="light"]) .buddy .bubble::before{{border-top-color:#a39e94}}}}
  :root[data-theme="dark"] .buddy .bubble{{background:#23242a;color:#ebe8e1;border-color:#a39e94}} :root[data-theme="dark"] .buddy .bubble::after{{border-top-color:#23242a}} :root[data-theme="dark"] .buddy .bubble::before{{border-top-color:#a39e94}}
  @media (prefers-reduced-motion: reduce){{.buddy .bot,.buddy .bubble{{opacity:1;transform:none;animation:none!important}} .buddy .wing,.buddy .bot svg,.buddy .eyes{{animation:none!important}}}}
  @media (max-width:760px){{
    .frame-wrap{{grid-template-columns:1fr}}
    .buddy{{position:static;order:-1;flex-direction:row;align-items:flex-end;justify-content:flex-end;margin:22px 0 -6px;padding:0 4px}}
    .buddy .bubble{{width:auto;max-width:520px;transform-origin:100% 80%}}
    .buddy .bot svg{{width:110px;height:100px}}
    .buddy .bubble::before,.buddy .bubble::after{{left:auto;margin-left:0;right:-10px;bottom:22px;border:8px solid transparent;border-left-color:#2f3340;border-right:0;border-top-color:transparent}}
    .buddy .bubble::after{{right:-7px;border-left-color:#fff;border-top-color:transparent}}
  }}

  /* ===== Browser-like frame around each embedded component ===== */
  .frame{{margin-top:20px;border:1px solid var(--rule);border-radius:12px;overflow:hidden;box-shadow:var(--shot-shadow);background:#f6f7fb}}
  .frame-bar{{display:flex;align-items:center;gap:12px;padding:9px 14px;background:var(--frame-chrome);color:var(--frame-chrome-fg);font:12.5px var(--sans)}}
  .fdots{{display:inline-flex;gap:6px}}
  .fdots i{{width:10px;height:10px;border-radius:50%;background:#d6d3cb;display:inline-block}}
  .fdots i:nth-child(1){{background:#f1655a}} .fdots i:nth-child(2){{background:#f4bf4f}} .fdots i:nth-child(3){{background:#61c554}}
  .furl{{flex:1;text-align:center;font-family:var(--mono);font-size:12px;opacity:.85}}
  .frame-bar.dark{{background:#0b0f1c;color:#9aa0b5}}
  .ftab{{padding:3px 10px;border-radius:6px;background:#161c30;color:#e4e8f5;font-family:var(--mono);font-size:12px}}
  .fpath{{margin-left:auto;font-family:var(--mono);font-size:11.5px;opacity:.7}}

  /* ===== Fake IDE ===== */
  .editor{{background:#0f1424;max-height:520px;overflow:auto}}
  .editor pre{{margin:0;padding:14px 18px 18px 0;font:13px/1.55 var(--mono);color:#d9def0;counter-reset:ln}}
  .editor code{{display:block;background:none;padding:0;font-size:inherit;white-space:pre}}
  .editor .ln{{display:inline-block;width:52px;padding-right:16px;text-align:right;color:#4a5170;user-select:none}}
  .editor .tok-keyword{{color:#c678dd;font-weight:600}} .editor .tok-builtin{{color:#56b6c2}} .editor .tok-string{{color:#98c379}}
  .editor .tok-comment{{color:#7c8095;font-style:italic}} .editor .tok-number{{color:#d19a66}} .editor .tok-function{{color:#61afef}} .editor .tok-decorator{{color:#d19a66}}

  /* ===== Scoped copy of the real app stylesheet (web_ui/static/styles.css) ===== */
  .app{{background:#f6f7fb;color:#1a1d29;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;color-scheme:light}}
  .app .topbar{{display:flex;justify-content:space-between;align-items:center;padding:12px 22px;background:#fff;border-bottom:1px solid #e7e9f1;flex-wrap:wrap;gap:8px}}
  .app .topbar-left{{display:flex;align-items:center;gap:10px}}
  .app .brand{{font-weight:600;letter-spacing:-0.01em;font-size:15px}}
  .app .dot{{width:10px;height:10px;background:linear-gradient(135deg,#6366f1,#06b6d4);border-radius:50%}}
  .app .topbar-right{{display:flex;align-items:center;gap:10px;font-size:13px}}
  .app .chip{{padding:3px 9px;background:#eef0f7;border-radius:999px;font-weight:500;font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:12px;color:#353846}}
  .app .chip.status{{background:#e6f4ff;color:#1d6fc5}}
  .app .chip.status[data-status="running"]{{background:#fff3d1;color:#8a6300}}
  .app .chip.status[data-status="awaiting"]{{background:#fde8eb;color:#b3344b}}
  .app .muted{{color:#6b7080}}
  .app .feed{{max-width:880px;margin:0 auto;padding:20px 20px 24px;display:flex;flex-direction:column;gap:14px}}
  .app .msg{{background:#fff;border:1px solid #e7e9f1;border-radius:12px;padding:16px 18px;animation:appear 220ms ease}}
  @keyframes appear{{from{{opacity:0;transform:translateY(4px)}}to{{opacity:1;transform:translateY(0)}}}}
  .app .msg-head{{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:10px}}
  .app .msg-head .label{{font-weight:600;font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:#4f5363}}
  .app .msg-body{{white-space:pre-wrap}}
  .app .md{{white-space:normal}} .app .md p{{margin:8px 0}} .app .md strong{{color:#1a1d29}}
  .app .msg-status a{{color:#4f46e5;font-weight:600}}
  .app .msg-status{{background:#f0f3fb;border:1px dashed #c8cdde;color:#4f5363;font-size:13.5px;padding:10px 14px;border-radius:10px}}
  .app .msg-q{{border:1px solid #cdd1ff;background:#fafbff}}
  .app .msg-q .label{{color:#4f46e5}}
  .app .msg-q .qstatus{{font-size:12px}}
  .app .msg-q .qstatus.answered{{color:#1a7f3c}}
  .app .q-prompt{{font-size:15px;margin-bottom:14px;color:#1a1d29}}
  .app .q-body{{display:flex;flex-direction:column;gap:10px}}
  .app .q-textarea{{width:100%;box-sizing:border-box;min-height:220px;padding:10px 12px;background:#fff;border:1px solid #d9dceb;border-radius:10px;font:13px/1.55 var(--mono);color:#1a1d29;resize:vertical}}
  .app .q-textarea.q-code{{min-height:320px;padding:14px 16px;background:#0f1424;color:#d9def0;border-color:#0f1424;white-space:pre;overflow:auto}}
  .app .q-prompt code{{font-size:13px}}
  .app .q-options{{display:grid;gap:8px}}
  .app .q-option{{display:flex;flex-direction:column;gap:4px;padding:12px 14px;background:#fff;border:1px solid #e7e9f1;border-radius:10px;cursor:pointer;transition:border-color 100ms ease,background 100ms ease;text-align:left;font:inherit;color:inherit}}
  .app .q-option:hover:not(:disabled){{border-color:#aeb4ff;background:#f4f5ff}}
  .app .q-option.selected{{border-color:#4f46e5;background:#eef0ff}}
  .app .q-option .opt-label{{font-weight:600;font-size:14px}}
  .app .q-option .opt-desc{{font-size:12.5px;color:#6b7080}}
  .app .q-option:disabled{{cursor:default;opacity:.6}}
  .app .q-actions{{display:flex;justify-content:flex-end;gap:8px}}
  .app .btn{{border:0;background:#4f46e5;color:#fff;font:inherit;font-weight:600;font-size:14px;padding:9px 16px;border-radius:8px;cursor:pointer;transition:background 100ms ease}}
  .app .btn:hover{{background:#4338ca}}
  .app .btn:disabled{{background:#b6b9d1;cursor:not-allowed}}
  .app .q-note{{margin-top:12px;padding:10px 14px;border-radius:10px;background:#eef0ff;border:1px solid #cdd1ff;font-size:13.5px;color:#2a2540}}
  .app .q-note a{{color:#4f46e5;font-weight:600}}
  .app code{{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;background:#eef0f7;color:#1a1d29;padding:1px 6px;border-radius:6px;font-size:13px}}
  .app .msg-cta{{background:#f0f1ff;border:1px solid #cdd1ff;display:flex;flex-direction:column;align-items:flex-start;gap:12px}}
  .app .msg-cta .cta-prompt{{font-size:14px}}
  .app .btn-cta{{background:#4f46e5;color:#fff;border:0;font:inherit;font-weight:600;font-size:14.5px;padding:11px 22px;border-radius:10px;cursor:pointer;transition:background 100ms ease,transform 80ms ease}}
  .app .btn-cta:hover:not(:disabled){{background:#4338ca;transform:translateY(-1px)}}
  .app .btn-cta:disabled{{background:#b6b9d1;cursor:not-allowed;transform:none}}
  .app .thinking{{display:flex;align-items:center;gap:10px;padding:8px 14px;color:#6b7080;font-size:13px;font-style:italic;align-self:flex-start}}
  .app .thinking[hidden],.app [hidden]{{display:none}}
  .app .spinner{{width:14px;height:14px;border-radius:50%;border:2px solid #d9dceb;border-top-color:#4f46e5;animation:spin .85s linear infinite;display:inline-block}}
  @keyframes spin{{to{{transform:rotate(360deg)}}}}
  .app .msg-cmp .label{{color:#4f46e5}}
  .app .cmp-details>summary{{cursor:pointer;list-style:none;justify-content:flex-start;margin-bottom:0}}
  .app .cmp-details[open]>summary{{margin-bottom:10px}}
  .app .cmp-details>summary::-webkit-details-marker{{display:none}}
  .app .cmp-details>summary::before{{content:"▸";color:#4f46e5;margin-right:8px;font-size:12px}}
  .app .cmp-details[open]>summary::before{{content:"▾"}}
  .app .cmp-chars-head{{margin:12px 0 10px;padding:10px 14px;background:#f5f3ff;border:1px solid #e9d5ff;border-left:3px solid #7c3aed;border-radius:9px;font-size:15.5px;font-weight:600;color:#2a2540}}
  .app .cmp-chars{{list-style:disc;margin:0;padding-left:22px;color:#2a2540}}
  .app .cmp-chars li{{margin:4px 0;font-size:13px;line-height:1.5}}
  .app .cmp-chars li::marker{{color:#7c3aed}}
  .app .cmp-chars strong{{color:#6d28d9}}
  .app .cmp-viz{{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px}}
  .app .cmp-viz figure{{margin:0;flex:1 1 280px;min-width:240px}}
  .app .cmp-viz video,.app .lot-row video{{width:100%;border-radius:8px;background:#000;display:block;aspect-ratio:1/1}}
  .app .cmp-viz figcaption,.app .lot-row figcaption{{font-size:12.5px;color:#6b7080;margin-top:4px}}
  .app .viz-preferred{{margin-top:4px;font-size:12px;font-weight:600;color:#1d6fc5;background:#e6f4ff;padding:4px 8px;border-radius:6px;display:inline-block}}
  .app .viz-prob{{margin-top:4px;font-size:12.5px;font-weight:600;color:#4f46e5;background:#eef0ff;padding:3px 8px;border-radius:6px;display:inline-block}}
  .app .viz-return{{margin-top:4px;font-size:12.5px;font-weight:600;color:#2a2540}}
  .app .cmp-more{{margin-top:12px}}
  .app .cmp-more>summary{{cursor:pointer;font-size:13px;font-weight:600;color:#4f46e5}}
  .app .lot-viz{{display:flex;flex-direction:column;gap:12px;margin-top:12px}}
  .app .lot-row{{display:flex;flex-wrap:wrap;gap:12px}}
  .app .lot-row figure{{margin:0;flex:1 1 280px;min-width:220px}}
  .app .lot-c figure{{flex:0 1 360px}}
  .app .lot-preferred{{margin-top:0;margin-bottom:-4px;align-self:flex-start}}
</style>
</head>
<body>
<div class="page">

<header class="site">
  <div class="text">
    <p class="crumb"><a href="./">Discordance-Based Reward Debugger</a> &nbsp;/&nbsp; Walkthrough</p>
    <p>Once you have run <code>/repair-setup &lt;env_name&gt;</code> and opened <b>http://127.0.0.1:8000</b>, everything below happens on that page. The panels on this page are the real interface, so click around.</p>
    <nav>
      <a href="#tour">Start the tour &darr;</a>
      <a href="./#setup">Set up</a>
      <a href="https://github.com/Stephanehk/Discordance-Based-Reward-Debugger">GitHub</a>
    </nav>
  </div>
</header>

<section id="tour">
  <ol class="tour">

    <li id="s2"><div>
      <div class="note">
        <h3>Write your reward function</h3>
        <p>The first thing you will see is this page. <b>Switch to your IDE and write your reward function in <code>workspaces/&lt;your_env_name&gt;/designed_reward_fn.py</code></b> inside the cloned repo.</p>
      </div>{frame(setup_feed('early'))}
    </div></li>

    <li id="s3"><div>
      <div class="note">
        <h3>Fill in the stub</h3>
        <p>Open the file in your IDE and fill in the body of the stub. <b>This is the only time you will need to write code.</b> Below is the reward function used in this walkthrough.</p>
      </div>{EDITOR}
    </div></li>

    <li id="s4"><div>
      <div class="note">
        <h3>Press “Identify possible flaws”</h3>
        <p>When you are done, return to your browser and press <b>Identify possible flaws</b>. You will be asked a few more quick questions, and then our system will spawn subagents to find flaws in your reward function.</p>
        <div class="callout"><b>This step can take up to an hour,</b> depending on the speed of your simulator. Feel free to come back to the page later, but don't close it!</div>
      </div>{frame(setup_feed('go'))}
    </div></li>

    <li id="s5"><div>
      <div class="note">
        <h3>Review the discordances</h3>
        <p>We will show you hypothesized reward misspecifications as <b>discordances</b>: comparisons on which our system hypothesizes your preference differs from the ranking your reward function induces. Most are trajectory pairs. <b>The left trajectory is always the one your reward function assigns higher return to.</b> The right trajectory is the one our system hypothesizes you might prefer instead. Each trajectory shows its return under your reward function, and underneath each pair is a one-line “why they disagree”. A dropdown shows how the two trajectories differ in more detail.</p>
        <p>The hypothesized misspecifications may not always be actual misspecifications! Expand the other discordances the system identified to see more.</p>
      </div>{with_buddy(frame(''.join(pair_card(i, i==1) for i in range(1,NPAIRS+1)), status='awaiting'), PAIR_NOTE)}
    </div></li>

    <li id="s6"><div>
      <div class="note">
        <h3>Review the lotteries</h3>
        <p>Some discordances are lotteries instead. A lottery asks: would you rather have the certain trajectory <b>C</b>, or a gamble that gives you trajectory <b>A</b> with one probability and trajectory <b>B</b> with the other? A badge marks which side your reward function assigns higher return to.</p>
      </div>{with_buddy(frame(''.join(lottery_card(i, i==1) for i in range(1,NLOTS+1)), status='awaiting'), LOT_NOTE)}
    </div></li>

    <li id="s7"><div>
      <div class="note">
        <h3>Decide what to do next</h3>
        <p>Below the discordances, the page asks you to pick one:</p>
      </div>{frame(QUESTION7)}
      <div class="note">
        <div class="aside">
          <b>Didn't find a flaw?</b> If none of the discordances identified a flaw in your reward function, you can run another round of this process. In our experience, after two more rounds (three rounds total) the system has covered many different hypothesized flaws.
        </div>
      </div>
    </div></li>

  </ol>
</section>

<section>
  <div class="text">
    <h2>Ready to try it?</h2>
    <p>Head back to the setup guide to get the tool running on your own environment.</p>
    <a class="cta" href="./#setup">Back to set up &rarr;</a>
  </div>
</section>

<footer>
  <div class="links">
    <a href="./">Home</a>
    <a href="https://github.com/Stephanehk/Discordance-Based-Reward-Debugger">GitHub</a>
  </div>
</footer>

</div>
{JS}
</body>
</html>
'''
OUT.write_text(page)
print('wrote', OUT, len(page))
