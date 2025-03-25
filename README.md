# Maintenance of this website

### General Notes

**It's worth familiarising yourself with Quarto and publishing web pages via the Quarto links
below, but here is a brief summary of how these web pages have been constructed.**

- Each web page is represented by a Quarto markdown file (.qmd) and this is where you edit the pages.
- The .qmd files get rendered as HTML files in the `docs` folder. These HTML files become the web pages.
- The `_quarto.yml` file contains the project settings, determining the layout and features of the website.
- Whenever you add a new page, you need to add the .qmd file to the `project: render:` section of the `_quarto.yml` file.
- The text that you see on each page is produced using standard markdown. Certain elements, such as the
table on the Schedule page have been created using Python code blocks.
- Once you have finished editing any of the files and you want to publish the updates, you need to run
`quarto render` in the terminal from the project directory root, then add and commit any updated files in Git before pushing
the updates to the GitHub repository.

### Updating the Schedule

- The Schedule is based on data held in the `schedule.csv` file in the data folder.
- The table that appears on the web page has been created using the `great_tables` Python library,
which has been used to format a `polars` dataframe of the .csv data.
- The `GitHub` column is to contain links to the relevant GitHub repository.
- If a recording has been made of a session, the link goes in the `Recording` column.
- Mark an "x" in the `Demonstration`, `Presentation` and `Notebook` columns depending on the content of the session.
These get converted to the relevant emojis in the Python code.

### Useful links

[Quarto](https://quarto.org/)

[Quarto publishing basics](https://quarto.org/docs/publishing/)

[Quarto publishing to GitHub Pages](https://quarto.org/docs/publishing/github-pages.html)
These pages have been published using method 1, via the `docs` directory.

[Emojis list](https://www.prosettings.com/emoji-list/) For the Schedule.

[Great Tables](https://posit-dev.github.io/great-tables/articles/intro.html) For the Schedule.

[Polars dataframes](https://docs.pola.rs/api/python/stable/reference/index.html) For the Schedule.