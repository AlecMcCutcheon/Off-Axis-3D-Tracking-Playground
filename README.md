# Off-Axis 3D Tracking Playground

A static web playground for off-axis rendering, face-driven camera parallax, hand pinch interaction, and room-bounded object physics.

## Why this exists

I kept seeing this style of demo on YouTube and in scattered clips online, but I could not find something easy to open and actually play with in my own hands.

I wanted to bring that experience to more people as a small, open playground: something you can run locally, move around in front of, drag objects with, and use as a starting point for your own ideas.

The MediaPipe face and hand tracking stack in this project genuinely blew my mind. It is not perfect and some areas are still rough, but it is fun and inspiring enough to share as-is.

## What you can do

- Move your head to drive the off-axis camera perspective
- Use hand tracking + pinch to grab and drag 3D objects
- Toggle gravity and tune gravity strength / bounce response
- Switch object collections (`Starter`, `Dense`, `Chaos`) in the settings panel
- Interact with room-bounded collisions (floor, walls, roof, front collision plane)

## Current rough edges

- Gravity behavior can still be unpredictable in edge-case contact stacks
- Resting stabilization is improved but still under active tuning
- Hand depth estimation can vary with lighting, camera quality, and pose
- This is a playground/demo, not a production physics engine

## Q&A

### "This does not look as good as what I've seen on YouTube. Why?"

That is fair feedback. While this effect is definitely possible, my implementation is not as polished as some of the best demos others have built. I am more than open to pull requests that improve this.

A big part of this is perception: it is much harder to trick your own eyes in person than it is to trick a camera recording. I have found that closing one eye makes the effect a bit more convincing.

If you want to see the effect better, hold your phone while moving around the scene and then play back the recording. It usually looks more convincing on video than in-person live viewing.

I am sure there are ways to make this feel more convincing in person (better graphics, tighter tuning, more refined movement), but your brain will still often catch that something is not quite right.

## Tech stack

- React + TypeScript + Vite
- Three.js + react-three-fiber + drei
- MediaPipe Tasks Vision (Face + Hand landmarker)

## Run locally

```bash
npm install
npm run dev
```

Open the local URL shown by Vite in your browser.

## Build static app

```bash
npm run build
npm run preview
```

The production-ready static files are output to `dist/`.
