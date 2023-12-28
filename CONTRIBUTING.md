# How to contribute to this repo?

1. Submit an issue to notify us with your plan in advance. 
2. Fork this repo to your own GitHub account. The detailed logistics to fork the repo, track the original repository as a remote of the fork, etc., can be found in this comprehensive guide by [Jake Jarvis](https://jarv.is/): [How To Fork a GitHub Repository & Submit a Pull Request](https://jarv.is/notes/how-to-pull-request-fork-github/)
3. Set this repo as the `upstream` remote and pull the latest commits to your own repo. 
4. Create a new branch from `master` and make changes there.
5. Before submitting a pull request, make sure you:
   1. no longer have new commits to push,
   2. run through all unit tests `pytest tests/` and make sure they all pass (if your changes cause some tests to fail, check and fix them), and
   3. have merged all commits from `master` of this repo onto your branch.
6. Please include the maintainers of this repo as reviewers of your pull request:
   - Clare Huang `@csyhuang`
   - Christopher Polster `@chpolste`

 This page is subject to changes. Suggestions on improvement are welcome.