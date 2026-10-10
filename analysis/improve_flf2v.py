"""Fix and tidy the FLF2V workflow: two real defects, then one place to drive it from.

The defects, both found by reading the graph rather than running it:
  1. #217 (the NEGATIVE encoder) held a verbatim copy of the 14k-character positive prompt, so the
     model was told to pursue and to avoid the same thing.
  2. the two source stills are 941x1672 (9:16) and the resize targeted 1280x720 (16:9) with a
     CENTRE crop, which throws away the top and bottom of a group shot - faces included.

Only widgets, titles, positions and one group box change. No link is touched, so the graph that
already runs keeps running.
"""
import json
import sys
from pathlib import Path

PROMPT = """A 10-second animated sequence created entirely within the exact same illustrated world shown in
the provided first and last reference frames.

The video begins exactly from the provided first frame and ends exactly on the provided final frame.

This is an animated illustration. It must NEVER become live action or photorealistic.


==================================================
ABSOLUTE CHARACTER CONTINUITY
==================================================

There are exactly FOUR people in the entire video.

The same four illustrated men shown in the reference images are the ONLY people who exist in this
scene. Never add a fifth person. Never duplicate, replace, merge, split or redesign any of them.
No person enters from off-screen. The four men keep the same arrangement: two standing slightly
behind, two in front.

Character 1 - front right: short light brown hair, clean-shaven, a warm crinkled smile,
cream white GALO sweatshirt.

Character 2 - front left: longer dark fringe, light moustache and soft beard,
black zip jacket worn open over a navy GALO sweatshirt.

Character 3 - back left: messy dark hair, strong pointed goatee, black GALO sweatshirt.

Character 4 - back right: swept silver-grey hair, heavy black-framed glasses, grey goatee,
black GALO sweatshirt.

Their identity, clothing, hairstyles, eyewear and body proportions remain unchanged throughout.


==================================================
ABSOLUTE STYLE CONTINUITY
==================================================

Clean stylized anime illustration. Crisp linework, flat cel shading, hard-edged shadows, warm
saturated colour, luminous lantern light. Illustrated faces, illustrated hands, illustrated food
and drink, illustrated steam.

NEVER become photorealistic. NEVER become live action. NEVER introduce realistic skin,
photographic food or photographic hands.


==================================================
0 - 3 SECONDS - OUTSIDE THE IZAKAYA
==================================================

Begin EXACTLY from the provided first frame: the four men posed together in front of the izakaya
at night, under the red lanterns and the white yakitori noren.

For the first moment they are almost still, the scene like a printed illustration coming to life.
The lanterns flicker gently. Warm light spills from the doorway. Faint steam drifts from inside.

Then Character 1 turns his head toward the entrance and gestures the others in with a small nod.
Character 2 laughs quietly. The two men behind begin to turn toward the door.

The camera holds, then begins a very slow push forward toward the entrance.


==================================================
3 - 6.5 SECONDS - STEPPING INSIDE
==================================================

The four men walk in through the noren one after another, the fabric brushing past their shoulders
and settling behind them.

The camera moves forward with them, passing under the noren into the warm interior: wooden
counter, rows of sake bottles, hanging paper lanterns, handwritten menu strips on the wall.

The light changes from the cool blue of the street to the deep amber of the interior.

They settle around a wooden table, lowering themselves onto their seats, still talking.


==================================================
6.5 - 9 SECONDS - THE TABLE FILLS
==================================================

The table comes alive: plates of yakitori skewers, a bowl of edamame, tamagoyaki, a ceramic sake
flask and small cups, a tall frosted glass of beer.

Thin illustrated steam rises from the hot food. The lanterns glow warmly behind them.

Character 2 lifts the beer mug. Character 1 raises a small sake cup. The two men behind lean in,
resting their arms on the shoulders of the two in front.

The movements are sequential, not simultaneous. Only one or two men move noticeably at a time.


==================================================
9 - 10 SECONDS - THE TOAST
==================================================

The composition settles EXACTLY into the provided final frame.

The four men raise their drinks together toward the camera for a toast, all four smiling, the beer
foam trembling slightly, the sake cup held steady.

Hold the final composition briefly.

The final frame must preserve exactly four characters, their arrangement, their final poses, the
full table of food and drink, the lantern-lit izakaya interior, and the original illustration style.


==================================================
CAMERA
==================================================

FRONTAL WIDE OUTSIDE -> SLOW PUSH THROUGH THE ENTRANCE -> SETTLE TO A FRONTAL TABLE COMPOSITION.

The camera stays centred on the four men. No lateral tracking, no orbit, no reverse angle, no
camera behind the characters, no 90-degree perspective change.


==================================================
MOTION
==================================================

Movement is restrained and sequential. Only one or two men perform a noticeable action at a time.
Keep hand movement slow, simple and anatomically stable.

Energy comes from the environment: flickering lanterns, drifting steam, the swaying noren, light
changing as they move inside.


==================================================
AUDIO
==================================================

There is NO MUSIC in this video. No soundtrack, no instrumental, no song, no melody, no beat,
no rhythm track, no background score.

The only sound is the natural, diegetic sound of the scene itself, recorded as if by a camera
standing with them:

the paper lanterns creaking faintly in the night air,
the noren curtain brushing past their shoulders as they step through,
footsteps on the floor,
the low murmur and clatter of an izakaya interior,
chairs shifting as they sit,
plates and small cups being set down on the wooden table,
yakitori sizzling,
sake pouring into a cup,
and their own quiet, wordless laughter.

On the final toast, the bright clink of a beer glass and a sake cup meeting.

No singing. No narration. No spoken dialogue. No chanting.


==================================================
NEVER
==================================================

more than four people, fewer than four people, a fifth person, background customers, staff,
people entering the frame, duplicate character, duplicate head, extra limbs, extra fingers,
morphing hands, character replacement, different clothing, different hairstyle, photorealism,
live action, real human faces, realistic skin, photographic food, 3D animation, style transition,
random text, camera orbit, side tracking, reverse angle, scene change to a different restaurant,
music, soundtrack, instrumental, song, melody, drums, bass guitar, background score, singing
"""

NEGATIVE = ("photorealistic, live action, real human face, realistic skin, photographic food, "
            "3D render, blurry, low quality, still frame, frozen, watermark, text overlay, "
            "subtitles, distorted face, distorted anatomy, extra fingers, extra limbs, deformed "
            "hands, duplicate person, fifth person, morphing, scene change, camera orbit")

NOTE = """# FLF2V — 最初と最後のフレームから動画を作る

**① 最初のフレーム** と **② 最後のフレーム** を入れ、**③ プロンプト** に間の出来事を書く。
LTX-2.5 は映像と音声を同時に作るので、音楽と効果音もプロンプトから生成される（無音にしたいなら
編集でミュート）。

| | 意味 |
|---|---|
| ④ 秒数 / fps | フレーム数 = 秒数 × fps + 1。10秒 × 24 = 241 フレーム |
| ⑤ 幅 / 高さ | **元画像と同じ比率にすること。** 違うと `center` で切り抜かれて顔が画面外に出る |
| ⑥ シード | randomize のままで毎回違う結果 |
| ⑦ ネガティブ | **短く。** ポジティブと同じ文章を入れると打ち消し合う |

## 調整の勘どころ

- **最初/最後のフレームへの吸着が弱い** → `LTXVAddGuide` 2つの `strength` を 0.7 から 0.85〜1.0 へ
- **動きが足りない** → プロンプトの動作を具体的に。`strength` を下げるのも効く（固定が緩む）
- **途中で人物が変わる** → キャラクター定義を特徴（眼鏡・髭・服の色）で書く。位置だけでは安定しない
- **尺を変える** → ④ の秒数のみ。15秒 = 361フレームで VRAM と時間が増える

サンプラーは 9 ステップ（`ManualSigmas`）。精細度を上げたい場合は2段構成にする手もある。
"""

CONTROLS = [
    # id, title, position, size
    (31, "① 最初のフレーム", [-1000, 40], [430, 580]),
    (39, "② 最後のフレーム", [-540, 40], [430, 580]),
    (252, "③ プロンプト（間に何が起きるか）", [-1000, 660], [970, 430]),
    (198, "④ 秒数", [-1000, 1120], [230, 110]),
    (205, "fps", [-750, 1120], [230, 110]),
    (215, "⑤ 幅", [-500, 1120], [230, 110]),
    (216, "⑤ 高さ", [-250, 1120], [230, 110]),
    (196, "⑥ シード", [-1000, 1260], [280, 110]),
    (217, "⑦ ネガティブ（短く）", [-700, 1260], [670, 200]),
]


def main(path):
    p = Path(path)
    wf = json.loads(p.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in wf["nodes"]}

    before_neg = len(str(nodes[217]["widgets_values"][0]))
    nodes[252]["widgets_values"][0] = PROMPT
    nodes[217]["widgets_values"][0] = NEGATIVE
    nodes[215]["widgets_values"][0] = 576
    nodes[216]["widgets_values"][0] = 1024
    nodes[68]["widgets_values"][0] = "video/GALO_izakaya_flf2v"

    for nid, title, pos, size in CONTROLS:
        nodes[nid].update(title=title, pos=pos, size=size)

    note = next((n for n in wf["nodes"] if n["type"] == "MarkdownNote"), None)
    note = json.loads(json.dumps(note))
    note.update(id=990, title="使い方", pos=[-1000, -420], size=[970, 430],
                widgets_values=[NOTE], mode=0)
    wf["nodes"] = [n for n in wf["nodes"] if n["id"] != 990] + [note]

    wf.setdefault("groups", [])
    wf["groups"] = [g for g in wf["groups"] if g.get("title") != "① 入力・設定"]
    wf["groups"].append({"id": 1, "title": "① 入力・設定（ここだけ操作すればOK）",
                         "bounding": [-1030, -500, 1030, 1990],
                         "color": "#3f789e", "font_size": 24, "flags": {}})

    p.write_text(json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"negative prompt: {before_neg} chars -> {len(NEGATIVE)} chars")
    print(f"output size    : 1280x720 -> {nodes[215]['widgets_values'][0]}"
          f"x{nodes[216]['widgets_values'][0]}")
    print(f"prompt         : {len(PROMPT)} chars written to #252")
    print(f"controls       : {len(CONTROLS)} nodes retitled and grouped, usage note added")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
