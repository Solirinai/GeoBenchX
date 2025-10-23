# GeoBenchX: Geospatial LLM Tool-Calling Benchmark

This repo contains the code and benchmark data from our paper comparing how well different LLMs handle geospatial tasks using function calling and suggesting LLM-as-Judge based evaluation framework to automate benchmarking of future releases of LLMs.

## What's this about?

We wanted to see how good current LLMs are at solving multi-step geospatial problems. These tasks are pretty unique because they need the model to:
- Know geographical facts
- Understand spatial data and coordinates  
- Work with georeferenced datasets
- Get what coordinate reference systems are
- Handle different types of geodata

## Experiment Setup

We built a simple ReAct agent using LangGraph with 23 specialized geospatial tools. The agent can:
- Read tabular and geospatial data (vector/raster formats)
- Perform dataframe operations (merge, filter, spatial joins)
- Do spatial analysis (buffers, overlaps, distance calculations)
- Create visualizations (choropleth maps, heatmaps, contour lines)

Each task gets up to 25 iterations of "LLM suggests tool → receives response" cycles. The agent works autonomously without Human-In-The-Loop and is explicitly told not to ask for clarifications or provide code solutions.

In addition, we included a `reject_task()` tool that lets the agent explicitly declare when a task can't be solved with the available datasets and tools. This is crucial for real-world applications to reduce hallucinated outputs.

For each task, we saved the complete conversation history between the LLM and tools as HTML files, including all plotted outputs, maps, and intermediate results ([example](https://htmlpreview.github.io/?https://github.com/Solirinai/GeoBenchX/blob/main/assets/Gemini_2.5_pro_TASK_250309_135125_275666.html)). This makes it easy to inspect the agent's reasoning process and debug issues.

## What we tested

**Models evaluated:**
- **Anthropic**: Claude Sonnet 3.5, Claude Sonnet 4, Claude Haiku 3.5  
- **Google**: Gemini 2.0 Flash, Gemini Pro 2.5 Preview  
- **OpenAI**: GPT-4o, GPT-4.1, o4-mini

Exact models releases [here.](https://github.com/Solirinai/GeoBenchX/blob/main/geobenchx/constants.py)

## Benchmark Dataset


**200+ tasks** across 4 complexity groups. Each group includes both **solvable** and **unsolvable** tasks to test whether models can recognize their limitations. Additionally, we included three tasks that don't require any tools or datasets (answerable from the LLM's world knowledge) for debugging purposes:

1. **Merge-visualize** (36 tasks) - Join statistical data with geography, create maps
   - *Example: "Map the relationship between GDP per capita and electric power consumption per capita globally"*

2. **Process-merge-visualize** (56 tasks) - Data processing + mapping  
   - *Example: "Compare rural population percentages in Western Asian countries"*

3. **Spatial operations** (53 tasks) - Spatial joins, buffers, distance calculations
   - *Example: "How many people live within 1 km from a railway in Bangladesh?"*

4. **Heatmaps & contour lines** (54 tasks) - Complex spatial analysis + specialized visualizations
   - *Example: "Create a heatmap of earthquake occurrences in regions with high population growth"*

**Data included:**
- 18 statistical datasets (economic indicators, emissions, health data)  
- 21 vector datasets (countries, cities, infrastructure, natural features)
- 11 raster datasets (snow cover, floods, population)

## Evaluation Method

We use an **LLM-as-Judge** approach with a panel of 3 judges (Claude Sonnet 3.5, GPT-4.1, Gemini 2.5 Pro Preview). Each task has manually annotated reference solutions, and judges score semantic equivalence between model outputs and ground truth.

**Scoring**: 
- 2: Match 
- 1: Partial match
- 0: No match

The evaluator judges were themselves tested against human annotations on 50 tasks, achieving 88-96% agreement with human scores.

## Key Results

**🏆 OpenAI's o4-mini came out on top** with the best overall performance across all task types. Its success comes from two key strengths:
- **Exceptional at spotting unsolvable tasks** (90% accuracy vs 55% for the second-best model)
- **Strong performance on solvable tasks** (second-best accuracy)

**🥈 Claude Sonnet 3.5 ranked second overall** by consistently placing in the top three across multiple categories. It showed the most balanced performance with equal accuracy on both solvable and unsolvable tasks.

**⚖️ Clear trade-offs emerged** among other top performers:
- **Claude Sonnet 4**: Best at solving tasks but struggles to identify unsolvable ones (3x better on solvable vs unsolvable)
- **GPT-4o & Gemini 2.5 Pro**: Better at rejecting impossible tasks than solving them (63% and 41% better respectively at rejection vs solving)

![Comparative performance of the benchmark set](/assets/LLMs%20performance%20by%20task%20groups.png)

## Why This Matters

This benchmark fills a gap in evaluating LLMs on domain-specific tasks that require both factual knowledge and procedural reasoning. Unlike general coding benchmarks, these tasks test whether models can navigate the complexities of real-world geospatial analysis workflows.

---

*See [our paper](https://arxiv.org/pdf/2503.18129) for full methodology details and results comparing model performance across different task types. This is the second version of the benchmark. At the previous interation, we benchmarked Sonnet 3.5 and 3.7, Haiku 3.5, Gemini 2.0, GPT-4o, GPT-4o mini, and o3-mini models (avaiable as [GeoBenchX: Benchmarking LLMs for Multistep Geospatial Tasks version 1](https://arxiv.org/pdf/2503.18129v1)).*


## This repository contains:
- [Tasks sets](https://github.com/Solirinai/GeoBenchX/tree/main/benchmark_set):
      1) Benchmark set of 202 geospatial tasks with reference solutions
      2) Set of tasks to tune evaluator agent - 50 tasks containing both reference and generated solutions where the generated solutions are scored manually. 
- [Link for datasets for the agent to solve the benchmarkset](https://github.com/Solirinai/GeoBenchX/tree/main/data)      
- [Modules and prompts](https://github.com/Solirinai/GeoBenchX/tree/main/geobenchx) for the task solving agent, evaluator agent, tools.
- [Notebooks](https://github.com/Solirinai/GeoBenchX/tree/main/notebooks) with code to generate geospatial tasks, tune evaluator agent if different judge panel is planned, to benchmark the evaluated LLMs on benchmark set.
- Folders for processing: to [save results](https://github.com/Solirinai/GeoBenchX/tree/main/results) and [scratch folder](https://github.com/Solirinai/GeoBenchX/tree/main/scratch) for tools.



## Citation

If you use this benchmark or code in your research, please cite:

BibTeX:
```bibtex
@misc{krechetova2025geobenchxbenchmarkingllmsmultistep,
      title={GeoBenchX: Benchmarking LLMs for Multistep Geospatial Tasks}, 
      author={Varvara Krechetova and Denis Kochedykov},
      year={2025},
      eprint={2503.18129},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/pdf/2503.18129}, 
}