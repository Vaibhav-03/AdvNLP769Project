"""
RAG Generator for TradingAgents

Generates enhanced responses by combining retrieved documents with LLM generation.
Supports different generation strategies for various trading contexts.
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI


class BaseGenerator(ABC):
    """Abstract base class for generators."""
    
    @abstractmethod
    def generate(self, query: str, retrieved_docs: List[Dict[str, Any]], **kwargs) -> str:
        """Generate response using query and retrieved documents."""
        pass


class TradingContextGenerator(BaseGenerator):
    """Generator specialized for trading contexts."""
    
    def __init__(self, llm, config: Dict[str, Any]):
        self.llm = llm
        self.config = config
    
    def generate(self, 
                query: str, 
                retrieved_docs: List[Dict[str, Any]], 
                context_type: str = "general",
                ticker: Optional[str] = None,
                trade_date: Optional[str] = None) -> str:
        """
        Generate trading-focused response using retrieved documents.
        
        Args:
            query: Original query
            retrieved_docs: Retrieved relevant documents
            context_type: Type of context ('market', 'fundamentals', 'news', 'social')
            ticker: Stock ticker symbol
            trade_date: Trading date
            
        Returns:
            Generated response
        """
        # Format retrieved documents
        context_docs = self._format_documents(retrieved_docs)
        
        # Get context-specific prompt
        system_prompt = self._get_system_prompt(context_type, ticker, trade_date)
        
        # Create user prompt with context
        user_prompt = self._create_user_prompt(query, context_docs, context_type)
        
        # Generate response
        messages = [
            ("system", system_prompt),
            ("human", user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def _format_documents(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents for context."""
        if not retrieved_docs:
            return "No relevant documents found."
        
        formatted_docs = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc['content'][:1000]  # Truncate long documents
            metadata = doc.get('metadata', {})
            doc_type = metadata.get('document_type', 'unknown')
            date = metadata.get('date', 'unknown')
            similarity = doc.get('similarity_score', 0.0)
            
            formatted_doc = f"""
Document {i} ({doc_type}, {date}, similarity: {similarity:.2f}):
{content}
"""
            formatted_docs.append(formatted_doc)
        
        return "\n".join(formatted_docs)
    
    def _get_system_prompt(self, context_type: str, ticker: Optional[str], trade_date: Optional[str]) -> str:
        """Get context-specific system prompt."""
        base_prompt = f"""You are an expert financial analyst with access to a comprehensive knowledge base of financial documents, market data, and trading information. Your role is to provide accurate, insightful analysis based on the retrieved documents and your expertise.

Current Analysis Context:
- Ticker: {ticker or 'Not specified'}
- Date: {trade_date or 'Not specified'}
- Context Type: {context_type}

Guidelines:
1. Base your analysis primarily on the retrieved documents
2. When documents are insufficient, clearly state limitations
3. Provide specific, actionable insights
4. Consider both quantitative and qualitative factors
5. Highlight key risks and opportunities
6. Maintain professional, objective tone
"""
        
        if context_type == "market":
            base_prompt += """
Market Analysis Focus:
- Technical indicators and price patterns
- Market trends and momentum
- Volume analysis
- Support and resistance levels
- Market sentiment indicators
"""
        elif context_type == "fundamentals":
            base_prompt += """
Fundamental Analysis Focus:
- Financial statement analysis
- Revenue and earnings trends
- Balance sheet strength
- Cash flow analysis
- Valuation metrics
- Industry comparisons
"""
        elif context_type == "news":
            base_prompt += """
News Analysis Focus:
- Recent news and events impact
- Market sentiment from news
- Regulatory developments
- Industry trends
- Company announcements
"""
        elif context_type == "social":
            base_prompt += """
Social Media Analysis Focus:
- Public sentiment and opinion
- Social media buzz and trends
- Influencer opinions
- Community discussions
- Sentiment indicators
"""
        
        return base_prompt
    
    def _create_user_prompt(self, query: str, context_docs: str, context_type: str) -> str:
        """Create user prompt with context."""
        prompt = f"""Query: {query}

Retrieved Context Documents:
{context_docs}

Please provide a comprehensive analysis based on the above query and retrieved documents. Focus on:
1. Key insights from the documents
2. How they relate to the query
3. Specific recommendations or conclusions
4. Any limitations or gaps in the information

Provide your analysis in a clear, structured format suitable for trading decisions.
"""
        return prompt


class RAGGenerator:
    """Main RAG generator that orchestrates different generation strategies."""
    
    def __init__(self, llm, config: Dict[str, Any]):
        """
        Initialize RAG generator.
        
        Args:
            llm: Language model instance
            config: Configuration dictionary
        """
        self.llm = llm
        self.config = config
        self.trading_generator = TradingContextGenerator(llm, config)
    
    def generate(self, 
                query: str, 
                retrieved_docs: List[Dict[str, Any]], 
                context_type: str = "general",
                ticker: Optional[str] = None,
                trade_date: Optional[str] = None,
                generation_strategy: str = "trading") -> str:
        """
        Generate response using RAG approach.
        
        Args:
            query: Original query
            retrieved_docs: Retrieved relevant documents
            context_type: Type of context
            ticker: Stock ticker symbol
            trade_date: Trading date
            generation_strategy: Strategy to use ('trading', 'general')
            
        Returns:
            Generated response
        """
        if generation_strategy == "trading":
            return self.trading_generator.generate(
                query=query,
                retrieved_docs=retrieved_docs,
                context_type=context_type,
                ticker=ticker,
                trade_date=trade_date
            )
        else:
            raise ValueError(f"Unknown generation strategy: {generation_strategy}")
    
    def generate_analysis_report(self, 
                               market_data: str,
                               news_data: str, 
                               fundamentals_data: str,
                               social_data: str,
                               ticker: str,
                               trade_date: str) -> str:
        """
        Generate comprehensive analysis report using RAG.
        
        Args:
            market_data: Market analysis data
            news_data: News analysis data
            fundamentals_data: Fundamentals analysis data
            social_data: Social media analysis data
            ticker: Stock ticker
            trade_date: Trading date
            
        Returns:
            Comprehensive analysis report
        """
        # Combine all data sources
        combined_context = f"""
Market Analysis:
{market_data}

News Analysis:
{news_data}

Fundamentals Analysis:
{fundamentals_data}

Social Media Analysis:
{social_data}
"""
        
        query = f"Provide a comprehensive trading analysis for {ticker} on {trade_date}"
        
        # Create mock retrieved docs for the combined context
        retrieved_docs = [{
            'content': combined_context,
            'metadata': {
                'document_type': 'combined_analysis',
                'ticker': ticker,
                'date': trade_date
            },
            'similarity_score': 1.0
        }]
        
        return self.generate(
            query=query,
            retrieved_docs=retrieved_docs,
            context_type="general",
            ticker=ticker,
            trade_date=trade_date
        )
