# Healthcare AI Analytics Assistant

This is a V1 portfolio project that demonstrates an end-to-end healthcare data analytics workflow using natural-language questions, SQL generation, pandas-based analysis and data visualization.

The application utilizes a TinyEHR SQLite healthcare database.

A user can ask questions such as:

  - What is the gender distribution of patients?
  - What is the average age of patients?
  - How many patients are there?
  - Give me patient age information.
  - Show admissions by year.
  - What is the age distribution?

Then the project starts its workflow.

### V1 Workflow

```text
User Question  
     ↓  
Context & Schema  
     ↓  
SQL Generation  
     ↓  
SQL Validation  
     ↓  
Query Execution  
     ↓  
Data Analysis  
     ↓  
Visualization  
```

### Project Components

The core Python packages for the analytics pipeline along with two ways to demo the application:

  - Streamlit App — interactive UI for a user to enter a question and render the results.
  - Jupyter Notebook — provides a demonstration of the full pipeline using a predefined question.
  - Source Packages — modular Python components for database access, SQL processing, analysis, and visualization.

The packages Structure:  
```text
┌─app
│	└── healthcare_analyzer_ui.py
│
├──notebooks
│	└── testing_full_pipeline.ipynb
│
└──src/
	├── analysis/
	│   ├── analyzer.py
	│   ├──	insights.py
	│   ├──	profiler.py	
	│   ├── statistics.py	
	│   └── types.py
	│	
	├── assistant/
	│   ├── assistant.py
	│   ├──	context.py
	│   ├──	conversation.py	
	│   ├── request_builder.py
	│   └── response.py
	│	
	├── db/
	│   ├── catalog.py
	│   ├──	database.py
	│   ├──	healthcare_terms.py	
	│   ├── schema.py
	│   ├── search.py
	│   ├── sqlite_database.py
	│   ├── synonyms.py	
	│   └── types.py
	│
	├── sql/
	│   ├── executor.py
	│   ├──	generator.py
	│   ├──	types.py	
	│   └── validator.py
	│ 	
	└── visualization/
		├── chart_builder.py
		├── chart_selector.py
		├── types.py
		└── visualizer.py
```

### Data

The project uses a TinyEHR SQLite database containing synthetic healthcare data.

The database file (tinyehr_mimic_format.db) is not included in this repository because its size exceeds GitHub's normal file-size limit.  

You can build a local SQLite database of tinyehr_mimic_format (see details at https://pypi.org/project/tinyehr/?#description).

### V1 Scope

This V1 version mainly focuses on demonstrating a reliable end-to-end healthcare analytics pipeline with some sample questions. 

The architecture is designed so that more additional features can be added in future versions.
