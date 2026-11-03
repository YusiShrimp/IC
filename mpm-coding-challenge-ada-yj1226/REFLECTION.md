# Reflection

Two short parts. Tick boxes by changing `[ ]` to `[x]`, and keep answers short: a line or two is plenty.

## Part 1: base version

Fill this in before you start Part 2 (about 2 minutes), then commit and push.

Which functions work (pass your tests)?

- [x] `lorenz96`
- [x] `life`, basic rules
- [ ] `life`, periodic boundaries
- [ ] `life`, two colours
- [ ] `life`, 3D

How confident are you that your base version is correct? (1 = not at all, 5 = very)

- Confidence:

Did you use any AI tools in Part 1? (That's fine: just say.)

- [ ] No
- [x] Yes: I asked the AI ​​to provide me with some ideas and some Python syntax I had forgotten.

## Part 2: AI-updated version

Fill this in before 17:00 (about 5 minutes), then commit and push.

Which AI tools did you use?

- [x] ChatGPT
- [ ] GitHub Copilot
- [ ] Claude
- [ ] Gemini
- [ ] Other:

What did you ask them to do?

- [x] Explain code or concepts
- [ ] Find or fix bugs
- [x] Speed up the code
- [x] Improve clarity or structure
- [ ] Write or improve tests
- [x] Finish functions I hadn't done in Part 1
- [ ] Other:

One suggestion you accepted, and why:

- I adopted NumPy vectorization for the Lorenz 96 update and SciPy convolution to count Game of Life neighbours. This avoids Python loops over individual cells and helps meet the runtime requirement for large grids.

One suggestion you rejected or changed, and why:

- Instead of having the AI ​​directly modify `automata.py`, I first completed and submitted Part 1, then saved the improvements in `automata_updated.py`, thereby preserving my original base version for comparison.

One time the AI was wrong or unhelpful, and how you noticed:

- It changed the formula of lorenz96 with different index.

How much did writing your own version first change the way you used the AI? (1 = not at all, 5 = a lot)

- Score: 5
- How: Since I had already written a basic version myself, I was able to compare the AI-generated code with my own implementation and note that Part 1 had already been submitted—meaning I was now working on Part 2. I will continue to verify whether the AI's implementation complies with the specifications, rather than simply checking if the code runs.

Is your AI-updated version better than your base version? In what way (speed, clarity, correctness, ...)?

- I would say definitely yes. The updated version supports periodic boundaries, two-colour Life and 3D Life. It took 6.76 seconds for the 1024×1024, 60-step 2D case and 5.33 seconds for the 101×101×101, 40-step 3D case, both under the 30-second limit. My base version took 0.671 seconds for 128×128 over 5 steps; scaling that result estimates about 515 seconds for the full 2D case, though this is only an estimate.

## Using your answers

- [ ] I'm happy for my answers in this file to be used, anonymised and combined with others', to help improve this
  module and plan future teaching.

Ticking this is optional, and doesn't affect your feedback either way.
