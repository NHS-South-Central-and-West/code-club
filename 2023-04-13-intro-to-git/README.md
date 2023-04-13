# Introduction to Git & GitHub

A Code Club talk given by Paul Johnson introducing the version control system Git and the Git-based version control and hosting service GitHub.

## What is Version Control?

Version control is a system that tracks and manages changes that are made to files, and gives the user the ability to revert to previous versions of a file if necessary.

It is a solution to the wide variety of problems you have probably encountered when working on projects where complexity, time, and collaboration is involved.

## What is Git?

Git is a distributed version control system, which means that every person working on a repository (project) has a copy of the entire repository that they have stored locally on their machine, to do with as they please. Once they have made changes that they want to share with the rest of the project team, they can push their changes to a remote repository, which is stored elsewhere (on platforms like GitHub, GitLab etc.) and can be accessed by anyone with the correct permissions.

This is very valuable in software development, and more broadly in any field that involves writing a lot of code.

## What Can Git Do For Me?

There are a myriad of potential benefits to using Git, but perhaps the most compelling argument in its favour is its ubiquity. Git is so ubiquitous across the software development and related industries that, in a community that is naturally predisposed to arguing about tools (and being contrarians), there is basically no one arguing that Git is not the right tool for the job. That alone is a pretty good signal of Git's value.

Here are some of the tangible benefits that have led to Git's widespread adoption:

### Security

- **No More Tears (Lost Work)** - Everyone has, at some point, lost work that they invested a lot of time and effort into. It is a lot easier to lose work if you are not using a version control system. Git protects you from computer mishaps, and even from your own mistakes (if you are rigorous about pushing changes).
- **Robust Best Practices** - Git is a _very_ widely used tool and it is a fundamental part of best practice in software development (and related fields). Incorporating Git into your workflow is a good way to ensure that you are following best practices around version control, and it sets your on the right path to wider adoption of best practices in software development.
- **Extra Layer of Protection** - Git is a great way to protect your code from potentially breaking changes. It's extremely easy to make a change to a file that causes a serious bug, either directly or indirectly. Git has a number of features that help mitigate this risk, from the fundamental idea of everyone developing on a local repository, to the use of "branches" (parallel versions of a repository) to isolate changes. In addition to this, platforms like GitHub have also developed a lot of features that help protect code, including CI/CD pipelines for testing code before it is merged into the main branch, and the ability to restrict who can merge code into the main branch.

### Productivity

- **Simple Collaborative Process** - Collaborating in a team, on a complex, interdependent project, is not easy. Git makes the process much simpler by allowing multiple people to work on the same codebase at the same time, and it makes the process of merging any changes as painless as possible.
- **Easy Experimentation** - Git allows you to maintain a stable production version of your codebase, while developing changes a "branch" of that codebase. Everyone can develop the project safe in the knowledge they are not going to destroy it if they do something dumb. This protects the project, and everyone working on it, but it also makes it a lot easier to experiment with new ideas and build new features without worrying that they will burn the whole thing to the ground.
- **Code-Focused Project Management** - Using Git with a platform like GitHub allows you to manage a project in a way that keeps the code at the centre of the process. You can track how a project develops over time, highlight issues, assign tasks, and plan development in a way that is easy to follow, and easy to frame in terms of the codebase itself.

### Openness

- **Code in the Open** - Platforms like GitHub & GitLab are a great way to share your work with the world, and to get inspiration & ideas from others. By fostering a culture of sharing our work and learning from each other, this creates an opportunity for everyone across the NHS to improve, and to maximise the value of the work we are doing. This is one of the central recommendations of the [Goldacre Review](https://www.goldacrereview.org/).
- **Public Code Portfolio** - Although it is not a requirement of the Git workflow that everything you do will be pushed to a public space for all to see, it is generally beneficial (assuming there are no information governance or security concerns around doing so), both to the NHS and SCW, and to developers themselves. By making your work public, you are, in effect, creating a portfolio of your work that you can use to demonstrate what you can do to prospective employers. Everyone's a winner!

## How Does One... Git Gud?

![A Knight saying "I Want You to Git Gud"](https://i.pinimg.com/564x/9f/72/8c/9f728cd259cff54742090dc9ab7363bb.jpg)

By using all the great resources below to learn Git & GitHub and, most of all, just **git**ting on with it. As with all things code (and probably all things everywhere), the fastest way to learn is to do. Try Git out, join the SCW GitHub organisation, and start contributing!

## Resources

### Learning Git

There are lots of resources that can help you learn Git. Rather than simply sharing those resources that have helped me, I will share a variety of approaches to teaching Git. Give them all a try, and see which one works best for you.

I would recommend the following resources for getting started:

- [NHS-R Git Training](https://github.com/nhs-r-community/git_training)
- [GitHub: Git Guides](https://github.com/git-guides)
- [Git Immersion](http://gitimmersion.com/)
- [The Odin Project: Git Basics](https://www.theodinproject.com/lessons/foundations-git-basics)
- [Learn Git Branching](https://learngitbranching.js.org/)

If you would prefer to work through a fully-fledged course instead of written guides, the following course (Udemy courses are not free but SCW has Udemy licenses) are a good start:

- [Udemy: Git Going Fast](https://www.udemy.com/course/git-going-fast/)
- [Udemy: The Git & GitHub Bootcamp](https://www.udemy.com/course/git-and-github-bootcamp)
- [Codecademy: Learn Git & GitHub](https://www.codecademy.com/learn/learn-git)

### Advanced Git

Having got to grips with the basics, I would recommend these resources for learning more advanced workflows in Git:

- [Git Documentation](https://git-scm.com/docs)
- [Pro Git](https://git-scm.com/book/en/v2)
- [Bitbucket: Advanced Git Tutorials](https://www.atlassian.com/git/tutorials/advanced-overview)
- [GitKraken: Git Tutorials](https://www.gitkraken.com/learn/git/tutorials)

### Cheatsheets

Finally, once you are comfortable using Git, you will continue to forget how to do simple things, because that's how the world works and there's nothing any of us can do about it. To help you in those moments, here are some cheatsheets provided by various Git platforms/clients (the cheatsheets are Git-specific, not platform- or client-specific)

- [GitHub Cheatsheet](https://training.github.com/downloads/github-git-cheat-sheet/)
- [Bitbucket Cheatsheet](https://www.atlassian.com/git/tutorials/atlassian-git-cheatsheet)
- [Git Tower Cheatsheet](https://www.git-tower.com/blog/git-cheat-sheet/)
- [GitKraken Cheatsheet](https://www.gitkraken.com/learn/git/commands)

### Using GitHub

Git & GitHub are different things, but GitHub **is** the most popular platform for hosting Git repositories, and for good reason. It's very good! There are lots of resources that can help you learn how to get the most out of GitHub:

- [GitHub Documentation](https://docs.github.com/en)
- [GitHub Guides](https://guides.github.com/)
- [GitHub Training Manual](https://githubtraining.github.io/training-manual)

## Your Time to Shine

Now go forth and...

![A gif of Spongebob flexing his muscles to reveal the words "Git" and "Gud" in his biceps](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExN2E3YmNjNWY1OTI1MTYyZmMzNDZlYTIzODlkOTMxYmU1M2YzYjQ2YiZjdD1n/10CopumcRWLMYM/giphy.gif)
