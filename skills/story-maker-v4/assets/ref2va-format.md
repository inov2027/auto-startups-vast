# MiniMax H3 Ref2VA format — V4 canonical contract

Use this exact ordered structure for every V4 generation prompt. It is designed
for MiniMax H3 reference generation and is validated before a paid render.

```text
subject_definitions:
<Subject 1> is ...
<Picture 1> is the storyboard-sheet reference; preserve its panel sequence and composition.

summary:
[reference generation] ...

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - ...
<Picture 1> (storyboard reference): fully_preserved - composition and panel sequence.

detailed_description:
<One or two visual-style sentences.>
[Shot 1] <visible action, camera motion, and sound.>
[Shot 2] At 00:03.000, <new visual information, action, camera motion, and sound.>
<Inline identity/count locks.>

overall_soundscape:
<Diegetic ambience, foley, and impacts.>

non_diegetic_music:
<Instrumentation, tempo, rhythm, and dynamics; or N/A.>
```

`[Shot 1]` carries no timestamp. Later shots use generation-local timestamps in
strictly increasing `At MM:SS.mmm` format. Each cut adds information; use a
camera move for framing-only change. Keep dialogue inside `<d>[Language] ...</d>`
and preserve the renderer attachment order: the storyboard is always `<Picture 1>`.
