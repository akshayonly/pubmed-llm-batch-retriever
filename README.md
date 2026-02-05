# PubMed Batch Fetcher

A Python-based tool for efficiently fetching and organizing biomedical literature from PubMed with intelligent token-aware batching.

## Features

- 🔍 **Flexible Search**: Supports keyword searches, MeSH terminology, and advanced PubMed queries
- 📊 **Token-Aware Batching**: Automatically splits articles into batches based on token limits (perfect for LLM context windows)
- 📅 **Date Filtering**: Retrieve articles from specific time ranges (e.g., last 5 years)
- 📝 **Review Filtering**: Option to fetch only review articles
- 📈 **Comprehensive Metrics**: Tracks words, tokens, and article statistics
- 💾 **Organized Output**: Clean text files with detailed logging and summaries
- ⚡ **Rate Limit Friendly**: Built-in delays to respect NCBI API guidelines

## Use Cases

- Literature reviews and systematic reviews
- Training data preparation for biomedical NLP models
- Research synthesis and meta-analysis
- Feeding article batches to LLMs for analysis
- Building domain-specific knowledge bases

## Key Capabilities

- Fetches titles and abstracts from PubMed/MEDLINE
- Uses tiktoken for accurate GPT-compatible token counting
- Handles large result sets (tested with 10,000+ articles)
- Provides detailed logs and batch distribution statistics
- Skips articles without abstracts automatically
```

### Tags/Keywords (for discoverability):
```
pubmed, bioinformatics, literature-mining, mesh, nlp, llm, research-automation, 
biomedical-informatics, text-mining, ncbi, entrez, scientific-literature
```

### One-liner (for social media/quick reference):
```
Fetch & batch PubMed articles with token limits — built for LLM-powered literature analysis

## License

This project is licensed under CC BY-NC 4.0 - see the [LICENSE](LICENSE) file for details.

**For commercial use**, please contact [your email] for permission.
