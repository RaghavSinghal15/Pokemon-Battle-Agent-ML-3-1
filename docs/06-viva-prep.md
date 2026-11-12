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

## Each owner must additionally defend their own module in depth

Expect "why did you do it this way and not that way?" for every design choice, which is why
choices go into [decisions.md](decisions.md) with their reasons as you make them.

## Questions we couldn't answer yet

Add any question that stumped someone in a mock viva, then add the answer once you have it.

| Question | Asked to | Answer |
|---|---|---|
| | | |
