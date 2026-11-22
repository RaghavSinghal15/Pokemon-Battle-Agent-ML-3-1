# Viva preparation

The examiner cross-questions hard and expects every member to understand the whole system, not
only their own part. Practise answering out loud, in your own words, without notes.

## Everyone must be able to answer

**The problem**
1. What does the agent do, in two sentences, to someone who has never played Pokémon?
2. Why is this harder than chess for a computer? (simultaneous moves, hidden information,
   randomness, size of the game)
3. Why doesn't "always use the strongest attack" work?
4. Why is this a machine learning problem and not just a set of rules?

**Data and leakage**
5. Where does the data come from, and how did you filter it? Why that rating threshold?
6. What is data leakage? Give a concrete example of how it could have happened in this project.
7. Why split by battle and not by turn? Why is the test set the newest battles?
8. Why do states contain only public information, even about our own team?

**Models**
9. What exactly does the opponent model predict, and why those classes?
10. Why logistic regression and boosted trees instead of a large neural network?
11. How did logistic regression compare with LightGBM, and why do you think that is?
12. What does calibration mean? How did you measure it, and why does the decision engine need it?
13. What is log loss and why report it alongside accuracy?
14. Which features mattered most? Does that make sense in game terms?

**Beliefs and decisions**
15. Explain the belief update with an example. Where is Bayes' rule?
16. What is in a cell of the payoff table, and what does it leave out?
17. What is the difference between best response and the minimax (LP) strategy? When should
    each one win?
18. Write down the linear program on the board and explain every constraint.
19. Why might a mixed (random) strategy ever be better than a fixed one?

**Results**
20. What are the win rates against each baseline, with their confidence intervals?
21. How do you know a difference between two versions is real and not luck?
22. What did each ablation show?
23. What didn't work, and why? Show a game the agent lost and explain the mistake.
24. What would you do next with more time?

**Version 2: what we added and why** (answers in [decisions.md](decisions.md) and
[the report](../reports/report.md))
25. 10% of human turns never show the choice. Why did dropping them bias the switch forecast,
    and how does a fourth "hidden" class fix it without a separate calibration step?
26. Why is log loss on visible turns only the wrong metric for choosing the opponent model?
    What does the all-turns log loss score on a hidden turn?
27. The forecast improved a lot with the belief threat features (D7), but the win rate barely
    moved. Why? What did that tell us to work on next?
28. How is a Swords Dance valued in the payoff table? Why is it worth nothing if we are about
    to be knocked out, and how does the code know that?
29. Why do entry hazards need their own valuation instead of the one-turn lookahead? Where does
    the hazard weight of 1.0 come from?
30. What is the value model, what is its label, and why can't the label ever be a feature?
31. Why score the payoff table by the change in win probability instead of HP? Give a game
    situation where 30% HP is worth very different amounts.
32. Why use the logistic-regression value model in the table when LightGBM had lower log loss?
33. Every row of a table must be measured from the same P(win now). What goes wrong if each
    row uses its own starting position?
34. What did the GRU over the whole game show, and why do boosted trees still win here?
    How do you know the GRU cannot see the future?
35. Why did the play-style feature make the forecast worse? Why cluster on five shares, not six?
36. How does the exact-move model work, and why score (position, move) pairs instead of one
    classifier over all moves?
37. What are the puzzles for, if we already have win rates? Which bug did they find?
38. How do you test on the public ladder without breaking the rules or publishing other
    players' names?

## Each owner must additionally defend their own module in depth

Expect "why did you do it this way and not that way?" for every design choice, which is why
choices go into [decisions.md](decisions.md) with their reasons as you make them.

## Questions we couldn't answer yet

Add any question that stumped someone in a mock viva, then add the answer once you have it.

| Question | Asked to | Answer |
|---|---|---|
| | | |
