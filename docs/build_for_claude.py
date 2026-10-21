"""Wrap docs/for_claude.md in the site's page chrome -> docs/for_claude.html.

The markdown file is the canonical copy (Claude Code is told to follow the
clone's copy once it has cloned); this page exists so the same text is
fetchable from the website. Re-run after editing for_claude.md.
"""
import html, pathlib, re

DOCS = pathlib.Path(__file__).resolve().parent
index = (DOCS / 'index.html').read_text()
style = re.search(r'<style>.*?</style>', index, re.S).group(0)
md = (DOCS / 'for_claude.md').read_text()

page = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Install with Claude Code</title>
<meta name="description" content="Step-by-step installation instructions for Claude Code to follow.">
{style}
<style>
  pre.doc{{white-space:pre-wrap;font-size:14.5px;line-height:1.55;margin-top:0}}
</style>
</head>
<body>
<div class="page">

<header class="site">
  <div class="text">
    <p class="crumb"><a href="./">Discordance-Based Reward Debugger</a> &nbsp;/&nbsp; Install with Claude Code</p>
    <p>This page is written for Claude Code to follow. In your project directory, open Claude Code and say:</p>
<pre id="prompt">Install the reward debugging tool from https://stephanehk.github.io/Discordance-Based-Reward-Debugger/for_claude.html</pre>
    <p class="muted" style="margin-top:14px">Prefer to do it by hand? See the <a href="manual_setup.html">manual setup</a>.</p>
  </div>
</header>

<section>
<pre class="doc">{html.escape(md)}</pre>
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
