"""
PubMed Batch Fetcher
A tool for fetching and organizing PubMed articles with token-aware batching.
"""

from Bio import Entrez
from datetime import datetime, timedelta
import time
import os
import argparse
import tiktoken


class PubMedFetcher:
    """
    Fetches PubMed articles and organizes them into token-limited batches.
    
    Attributes:
        query (str): PubMed search query (supports MeSH terms)
        years (int): Number of years to search back
        max_results (int): Maximum number of articles to fetch
        token_limit (int): Maximum tokens per batch
        output_dir (str): Directory for output files
        review_only (bool): Whether to fetch only review articles
    """
    
    def __init__(self, email, api_key, query, years=5, max_results=100, 
                 token_limit=8000, output_dir="./pubmed_output", review_only=False):
        """
        Initialize the PubMed fetcher.
        
        Args:
            email (str): Email for NCBI Entrez (required)
            api_key (str): NCBI API key (optional but recommended)
            query (str): PubMed search query
            years (int): Date range in years (default: 5)
            max_results (int): Maximum results to fetch (default: 100)
            token_limit (int): Token limit per batch (default: 8000)
            output_dir (str): Output directory path
            review_only (bool): Fetch only review articles (default: False)
        """
        # Set Entrez credentials
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
        
        self.query = query
        self.years = years
        self.max_results = max_results
        self.token_limit = token_limit
        self.output_dir = output_dir
        self.review_only = review_only

        # Initialize tokenizer for GPT-4/3.5-turbo
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Statistics
        self.total_articles = 0
        self.total_words = 0
        self.total_tokens = 0
        self.skipped_articles = 0
        self.batches = []

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize log file
        self.log_file = os.path.join(output_dir, "fetch_log.txt")
        self._log(f"PubMed Fetcher initialized at {datetime.now()}")
        self._log(f"Query: {query}")
        self._log(f"Date range: Last {years} year(s)")
        self._log(f"Max results: {max_results}")
        self._log(f"Review articles only: {review_only}")
        self._log(f"Token limit per batch: {token_limit}")

    def _log(self, message):
        """Log messages to file and console."""
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{message}\n")

    def search_pubmed(self):
        """
        Search PubMed and return list of PMIDs.
        
        Returns:
            list: List of PubMed IDs (PMIDs)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.years)

        mindate = start_date.strftime("%Y/%m/%d")
        maxdate = end_date.strftime("%Y/%m/%d")

        # Add review filter if requested
        search_query = self.query
        if self.review_only:
            search_query += ' AND "review"[Publication Type]'

        self._log("\n" + "="*60)
        self._log("SEARCHING PUBMED...")
        self._log("="*60)
        self._log(f"Effective query: {search_query}")

        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=search_query,
                mindate=mindate,
                maxdate=maxdate,
                datetype="pdat",
                retmax=self.max_results,
                usehistory="y"
            )

            record = Entrez.read(handle)
            handle.close()

            pmid_list = record["IdList"]
            total_count = int(record["Count"])

            self._log(f"Total articles found: {total_count}")
            self._log(f"Retrieving: {len(pmid_list)} articles")

            return pmid_list

        except Exception as e:
            self._log(f"ERROR during search: {e}")
            return []

    def fetch_articles(self, pmid_list):
        """
        Fetch article details (title and abstract only).
        
        Args:
            pmid_list (list): List of PubMed IDs
            
        Returns:
            list: List of article dictionaries with pmid, title, and abstract
        """
        articles = []
        batch_size = 50  # Fetch 50 articles at a time

        self._log("\n" + "="*60)
        self._log("FETCHING ARTICLE DETAILS...")
        self._log("="*60)

        for i in range(0, len(pmid_list), batch_size):
            batch_pmids = pmid_list[i:i + batch_size]

            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    id=",".join(batch_pmids),
                    rettype="xml",
                    retmode="xml"
                )

                records = Entrez.read(handle)
                handle.close()

                for article_record in records['PubmedArticle']:
                    try:
                        pmid = str(article_record['MedlineCitation']['PMID'])

                        # Extract title
                        title = article_record['MedlineCitation']['Article']['ArticleTitle']

                        # Extract abstract
                        abstract_data = article_record['MedlineCitation']['Article'].get('Abstract')
                        if abstract_data and 'AbstractText' in abstract_data:
                            abstract_parts = abstract_data['AbstractText']
                            if isinstance(abstract_parts, list):
                                abstract = " ".join([str(part) for part in abstract_parts])
                            else:
                                abstract = str(abstract_parts)
                        else:
                            self._log(f"WARNING: No abstract for PMID {pmid} - skipping")
                            self.skipped_articles += 1
                            continue

                        articles.append({
                            'pmid': pmid,
                            'title': title,
                            'abstract': abstract
                        })

                    except KeyError as e:
                        self._log(f"WARNING: Missing data for article - {e}")
                        self.skipped_articles += 1
                        continue

                # Progress update
                processed = min(i + batch_size, len(pmid_list))
                self._log(f"Processed {processed}/{len(pmid_list)} articles")

                # Be nice to NCBI servers
                time.sleep(0.34)

            except Exception as e:
                self._log(f"ERROR fetching batch {i//batch_size + 1}: {e}")
                continue

        self._log(f"\nSuccessfully fetched: {len(articles)} articles")
        self._log(f"Skipped (no abstract): {self.skipped_articles} articles")

        return articles

    def count_tokens(self, text):
        """Count tokens using tiktoken."""
        return len(self.tokenizer.encode(text))

    def count_words(self, text):
        """Count words in text."""
        return len(text.split())

    def process_and_batch_articles(self, articles):
        """
        Process articles and create token-limited batches.
        
        Args:
            articles (list): List of article dictionaries
        """
        self._log("\n" + "="*60)
        self._log("PROCESSING ARTICLES AND CREATING BATCHES...")
        self._log("="*60)

        current_batch = []
        current_batch_tokens = 0
        batch_number = 1

        for idx, article in enumerate(articles, 1):
            # Combine title and abstract
            combined_text = f"{article['title']} {article['abstract']}"

            # Count words and tokens
            word_count = self.count_words(combined_text)
            token_count = self.count_tokens(combined_text)

            # Update totals
            self.total_words += word_count
            self.total_tokens += token_count

            # Store article with metrics
            article['word_count'] = word_count
            article['token_count'] = token_count

            # Check if adding this article exceeds token limit
            if current_batch_tokens + token_count > self.token_limit and current_batch:
                # Save current batch
                self._save_batch(current_batch, batch_number, current_batch_tokens)
                self.batches.append({
                    'batch_number': batch_number,
                    'article_count': len(current_batch),
                    'token_count': current_batch_tokens
                })

                # Start new batch
                current_batch = [article]
                current_batch_tokens = token_count
                batch_number += 1
            else:
                # Add to current batch
                current_batch.append(article)
                current_batch_tokens += token_count

            # Progress update
            if idx % 10 == 0:
                self._log(f"Processed {idx}/{len(articles)} articles")

        # Save final batch
        if current_batch:
            self._save_batch(current_batch, batch_number, current_batch_tokens)
            self.batches.append({
                'batch_number': batch_number,
                'article_count': len(current_batch),
                'token_count': current_batch_tokens
            })

        self.total_articles = len(articles)

    def _save_batch(self, batch, batch_number, token_count):
        """Save a batch of articles to a text file."""
        filename = os.path.join(self.output_dir, f"pubmed_batch_{batch_number}.txt")

        with open(filename, 'w', encoding='utf-8') as f:
            for article in batch:
                f.write(f"PMID: {article['pmid']}\n\n")
                f.write(f"TITLE: {article['title']}\n\n")
                f.write(f"ABSTRACT:\n{article['abstract']}\n\n")
                f.write("-" * 40 + "\n\n")

        self._log(f"Saved batch {batch_number}: {len(batch)} articles, "
                 f"{token_count} tokens -> {filename}")

    def save_summary(self):
        """Save summary statistics to file."""
        summary_file = os.path.join(self.output_dir, "summary.txt")

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("PUBMED FETCH SUMMARY\n")
            f.write("="*60 + "\n\n")

            f.write(f"Search Query: {self.query}\n")
            f.write(f"Review Articles Only: {self.review_only}\n")
            f.write(f"Date Range: Last {self.years} year(s)\n")
            f.write(f"Fetch Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("-"*60 + "\n")
            f.write("FETCH STATISTICS\n")
            f.write("-"*60 + "\n")
            f.write(f"Total articles fetched: {self.total_articles}\n")
            f.write(f"Articles skipped (no abstract): {self.skipped_articles}\n")
            f.write(f"Total batches created: {len(self.batches)}\n\n")

            f.write("-"*60 + "\n")
            f.write("TOKEN & WORD METRICS\n")
            f.write("-"*60 + "\n")
            f.write(f"Total words: {self.total_words:,}\n")
            f.write(f"Total tokens: {self.total_tokens:,}\n")
            if self.total_articles > 0:
                f.write(f"Average words per article: {self.total_words // self.total_articles}\n")
                f.write(f"Average tokens per article: {self.total_tokens // self.total_articles}\n\n")

            f.write("-"*60 + "\n")
            f.write("BATCH DISTRIBUTION\n")
            f.write("-"*60 + "\n")
            for batch in self.batches:
                f.write(f"Batch {batch['batch_number']}: {batch['article_count']} articles, "
                       f"{batch['token_count']:,} tokens\n")

            f.write("\n" + "-"*60 + "\n")
            f.write("OUTPUT FILES\n")
            f.write("-"*60 + "\n")
            f.write(f"Output directory: {self.output_dir}\n")
            for i in range(1, len(self.batches) + 1):
                f.write(f"  - pubmed_batch_{i}.txt\n")
            f.write(f"  - summary.txt\n")
            f.write(f"  - fetch_log.txt\n")

        self._log(f"\nSummary saved to: {summary_file}")

    def print_summary(self):
        """Print summary to console."""
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"Query: {self.query}")
        print(f"Review only: {self.review_only}")
        print(f"Total articles: {self.total_articles}")
        print(f"Skipped: {self.skipped_articles}")
        print(f"Total batches: {len(self.batches)}")
        print(f"Total words: {self.total_words:,}")
        print(f"Total tokens: {self.total_tokens:,}")
        if self.total_articles > 0:
            print(f"Avg tokens/article: {self.total_tokens // self.total_articles}")
        print(f"\nOutput directory: {self.output_dir}")
        print("="*60)

    def run(self):
        """Main execution flow."""
        # Step 1: Search PubMed
        pmid_list = self.search_pubmed()
        if not pmid_list:
            self._log("No results found. Exiting.")
            return

        # Step 2: Fetch articles
        articles = self.fetch_articles(pmid_list)
        if not articles:
            self._log("No articles fetched. Exiting.")
            return

        # Step 3: Process and batch articles
        self.process_and_batch_articles(articles)

        # Step 4: Save summary
        self.save_summary()

        # Step 5: Print final summary
        self.print_summary()


def main():
    """Command-line interface for PubMed Batch Fetcher."""
    parser = argparse.ArgumentParser(
        description='Fetch PubMed articles with token-based batching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search
  python pubmed_fetcher.py --email you@email.com --query "machine learning" --years 5

  # MeSH term search
  python pubmed_fetcher.py --email you@email.com --query '"Deep Learning"[MeSH]' --years 3

  # Review articles only
  python pubmed_fetcher.py --email you@email.com --query "bioinformatics" --review-only

  # Custom output and limits
  python pubmed_fetcher.py --email you@email.com --query "CRISPR" --max-results 500 --token-limit 10000 --output-dir ./my_output
        """
    )
    
    parser.add_argument('--email', type=str, required=True, 
                       help='Your email (required by NCBI)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='NCBI API key (optional but recommended)')
    parser.add_argument('--query', type=str, required=True, 
                       help='PubMed search query (supports MeSH terms)')
    parser.add_argument('--years', type=int, default=5, 
                       help='Date range in years (default: 5)')
    parser.add_argument('--max-results', type=int, default=100, 
                       help='Maximum results to fetch (default: 100)')
    parser.add_argument('--token-limit', type=int, default=8000, 
                       help='Token limit per batch (default: 8000)')
    parser.add_argument('--output-dir', type=str, default='./pubmed_output', 
                       help='Output directory (default: ./pubmed_output)')
    parser.add_argument('--review-only', action='store_true', 
                       help='Fetch only review articles')

    args = parser.parse_args()

    fetcher = PubMedFetcher(
        email=args.email,
        api_key=args.api_key,
        query=args.query,
        years=args.years,
        max_results=args.max_results,
        token_limit=args.token_limit,
        output_dir=args.output_dir,
        review_only=args.review_only
    )

    fetcher.run()


if __name__ == "__main__":
    main()
