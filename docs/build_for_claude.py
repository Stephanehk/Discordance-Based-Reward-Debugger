"""Build docs/for_claude.html, the bootstrap page Claude Code fetches.

The page deliberately holds only the clone command and a pointer to
docs/for_claude.md (the complete instructions), because fetch tools
summarize pages and would drop the details. Re-run after editing index.html's
styles.
"""
import pathlib, re

DOCS = pathlib.Path(__file__).resolve().parent
index = (DOCS / 'index.html').read_text()
style = re.search(r'<style>.*?</style>', index, re.S).group(0)

page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Install with Claude Code</title>
<meta name="description" content="Step-by-step installation instructions for Claude Code to follow.">
{style}
<style>
</style>
</head>
<body>
<div class="page">

<header class="site">
  <div class="text">
    <p class="crumb"><a href="./">Discordance-Based Reward Debugger</a> &nbsp;/&nbsp; Install with Claude Code</p>
    <p>This page is written for Claude Code to follow. In your project directory, open Claude Code and say:</p>
<pre id="prompt">Install the reward debugging tool from https://stephanehk.github.io/Discordance-Based-Reward-Debugger/for_claude.html for task [task_name]</pre>
    <p class="muted" style="margin-top:14px">Replace <code>[task_name]</code> with the name of the task in your environment, e.g. <code>drawer-open</code> or <code>hammer</code>.</p>
    <p class="muted" style="margin-top:14px">Prefer to do it by hand? See the <a href="manual_setup.html">manual setup</a>.</p>
  </div>
</header>

<section>
  <div class="text">
    <h2>Claude: do this</h2>
    <p>Tools that fetch this page may summarize it, so the complete instructions live in the repository, not here. Run these two commands in the user's project directory, then read the file and follow it step by step:</p>
<pre>git clone https://github.com/Stephanehk/Discordance-Based-Reward-Debugger.git
cat Discordance-Based-Reward-Debugger/docs/for_claude.md</pre>
    <p>If the clone already exists, skip the clone and just read the file. The task named in the user's request is the one to set up. The file tells you how to find that task's environment and reward function, confirm both with the user, set up visualization, start the tool, and then run it from the same session.</p>
    <p class="muted">Humans can read the same file on GitHub: <a href="https://github.com/Stephanehk/Discordance-Based-Reward-Debugger/blob/main/docs/for_claude.md">docs/for_claude.md</a>.</p>
  </div>
</section>

<footer>
  <div class="links">
    <a href="./">Home</a>
    <a href="manual_setup.html">Manual setup</a>
    <a href="walkthrough.html">Walkthrough</a>
    <a href="https://github.com/Stephanehk/Discordance-Based-Reward-Debugger">GitHub</a>
  </div>
</footer>

</div>
</body>
</html>
'''
(DOCS / 'for_claude.html').write_text(page)
print('wrote', DOCS / 'for_claude.html', len(page))
