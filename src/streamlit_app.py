import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import sys
import os
import json, re
from typing import List, Dict, Any

# Import from the existing application
from utils.analysts import ANALYST_ORDER
from llm.models import LLM_ORDER, get_model_info
from main import run_hedge_fund
from utils.stock_charts_fixed import get_stock_chart_with_indicators



def format_reasoning_as_markdown(reasoning: Any) -> str:
    """
    Convert reasoning data into a Markdown string, with each metric on its own line.
    """

    # 1) Parse dictionary or JSON string
    if isinstance(reasoning, dict):
        data = reasoning
    else:
        try:
            data = json.loads(reasoning)
        except (ValueError, TypeError):
            return str(reasoning)

    if not isinstance(data, dict):
        return str(reasoning)

    # 2) Build a Markdown string
    lines = []
    for key, val in data.items():
        category = key.replace("_signal", "").replace("_", " ").title()

        if isinstance(val, dict):
            signal_type = val.get('signal', '').title()
            details_str = val.get('details', '')
            details_str = re.sub(r"\s+", " ", details_str).strip()
            detail_items = details_str.split(", ")

            # Heading
            lines.append(f"<h6 style='padding:0'>{category}</h6> ({signal_type})  ")
            # Bullets
            for item in detail_items:
                lines.append(f"- {item}")
            lines.append("")  # blank line
        else:
            # Simple string
            val = re.sub(r"\s+", " ", str(val)).strip()
            lines.append(f"<h6 style='margin:0'>{category}</h6>  - {val}  ")

    # 3) Return one big Markdown block
    return "  ".join(lines)











def format_reasoning_as_html(reasoning: Any) -> str:
    """
    Convert reasoning data into a bulleted HTML string.

    This function handles both JSON strings and Python dictionaries from the 'reasoning' field
    and converts them into a structured HTML format with categories and bullet points.
    If the input is neither a valid JSON string nor a dictionary, it returns the input as-is.

    Args:
        reasoning: A string containing JSON-formatted reasoning details or a Python dictionary.

    Returns:
        A string containing the reasoning details formatted as HTML.
    """
    # Handle the case where reasoning is already a dictionary
    data = None

    if isinstance(reasoning, dict):
        data = reasoning
    else:
        # Attempt to parse JSON if it's a string
        try:
            data = json.loads(reasoning) if isinstance(reasoning, str) else None
        except (ValueError, TypeError):
            # If not valid JSON, just return the raw input
            return str(reasoning)

    # If we couldn't get a dictionary, return the original input
    if not isinstance(data, dict):
        return str(reasoning)

    # Build HTML output
    html_blocks = []
    for key, val in data.items():
        # Example: key = "profitability_signal"
        category = key.replace("_signal", "").replace("_", " ").title()
        # Example: "profitability_signal" -> "Profitability"

        # Handle both string values and nested dictionaries
        if isinstance(val, dict):
            signal_type = val.get('signal', '').title()
            # Example: "bullish" -> "Bullish"
            details_str = val.get('details', '')
            # Example: "ROE: 145.30%, Net Margin: 24.30%..."

            # Convert the details to a bullet list
            detail_items_html = ""
            for detail_item in details_str.split(","):
                detail_items_html += f"<li>{detail_item.strip()}</li>"

            # One section per signal
            html_block = f"""
            <p><strong>{category}</strong> ({signal_type})</p>
            <ul>
                {detail_items_html}
            </ul>
            """
        else:
            # Handle simple string values
            html_block = f"""
            <p><strong>{category}</strong></p>
            <ul>
                <li>{val}</li>
            </ul>
            """

        html_blocks.append(html_block)

    # Join everything into one HTML string
    return "".join(html_blocks)



# Set page configuration
st.set_page_config(
    page_title="InvestAI Analyst Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .bullish {
        color: #4CAF50;
        font-weight: bold;
    }
    .bearish {
        color: #F44336;
        font-weight: bold;
    }
    .neutral {
        color: #FFC107;
        font-weight: bold;
    }
    .action-buy {
        color: #4CAF50;
        font-weight: bold;
    }
    .action-sell {
        color: #F44336;
        font-weight: bold;
    }
    .action-hold {
        color: #FFC107;
        font-weight: bold;
    }
    .action-short {
        color: #F44336;
        font-weight: bold;
    }
    .action-cover {
        color: #4CAF50;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    '<div class="main-header">InvestAI Analyst Platform</div>', unsafe_allow_html=True
)
st.markdown(
    "Make smarter trading decisions powered by AI analysts and portfolio management"
)

# Sidebar for inputs
with st.sidebar:
    st.markdown('<div class="sub-header">Configuration</div>', unsafe_allow_html=True)

    # Ticker input
    ticker_input = st.text_input(
        "Enter stock tickers (comma-separated)", "AAPL,MSFT,GOOGL"
    )
    tickers = [ticker.strip() for ticker in ticker_input.split(",")]

    # Date range selection
    st.markdown("### Date Range")
    end_date = st.date_input("End Date", datetime.now())
    start_date_options = {
        "1 Month": end_date - relativedelta(months=1),
        "3 Months": end_date - relativedelta(months=3),
        "6 Months": end_date - relativedelta(months=6),
        "1 Year": end_date - relativedelta(years=1),
        "Custom": "custom",
    }
    date_range = st.selectbox("Select Date Range", list(start_date_options.keys()))

    if date_range == "Custom":
        start_date = st.date_input("Start Date", end_date - relativedelta(months=3))
    else:
        start_date = start_date_options[date_range]

    # Portfolio settings
    st.markdown("### Portfolio Settings")
    initial_cash = st.number_input(
        "Initial Cash", min_value=1000.0, value=100000.0, step=1000.0
    )
    # Hide margin requirement but keep it as a variable with default value
    margin_requirement = 0.0

    # Analyst selection
    analyst_options = {display: value for display, value in ANALYST_ORDER}

    # Group analysts by type
    investor_analysts = [
        "Ben Graham",
        "Bill Ackman",
        "Cathie Wood",
        "Charlie Munger",
        "Phil Fisher",
        "Stanley Druckenmiller",
        "Warren Buffett",
    ]
    technical_analysts = [
        "Technical Analyst",
        "Fundamentals Analyst",
        "Sentiment Analyst",
        "Valuation Analyst",
    ]

    # Create two separate sections with their own radio buttons
    st.markdown("### AI Team")
    team_options = ["None"] + investor_analysts
    team_options = [a for a in team_options if a == "None" or a in analyst_options]

    selected_team_member = st.radio(
        "Select a team member (optional):", team_options, index=0  # Default to "None"
    )

    st.markdown("### AI Technicals")
    st.write("Select which technical analysts to include:")

    # Create checkboxes for each technical analyst
    selected_technicals = []
    for tech_analyst in technical_analysts:
        if tech_analyst in analyst_options:
            if st.checkbox(tech_analyst, value=True):  # Default to selected
                selected_technicals.append(tech_analyst)

    # Determine the final selected analyst(s)
    selected_analyst_name = "None"
    if selected_team_member != "None":
        selected_analyst_name = selected_team_member

    # Convert selection to the format expected by the backend
    selected_analysts = []

    # Add the selected team member if any
    if selected_analyst_name != "None" and selected_analyst_name in analyst_options:
        selected_analysts.append(analyst_options[selected_analyst_name])

    # Add all selected technical analysts
    for tech_analyst in selected_technicals:
        if tech_analyst in analyst_options:
            selected_analysts.append(analyst_options[tech_analyst])

    # Model selection - fixed to GPT-mini for now
    st.markdown("### LLM Model")

    # Find the GPT-mini model in the available models
    gpt_mini_model = None
    gpt_mini_provider = None

    # Look for a model with "mini" in the name from OpenAI
    for display, value, provider in LLM_ORDER:
        if "mini" in display.lower() and "openai" in display.lower():
            gpt_mini_model = value
            gpt_mini_provider = provider
            gpt_mini_display = display
            break

    # If not found, use the first OpenAI model as fallback
    if not gpt_mini_model:
        for display, value, provider in LLM_ORDER:
            if "openai" in display.lower():
                gpt_mini_model = value
                gpt_mini_provider = provider
                gpt_mini_display = display
                break

    # If still not found, use the first model in the list
    if not gpt_mini_model and LLM_ORDER:
        gpt_mini_display, gpt_mini_model, gpt_mini_provider = LLM_ORDER[0]

    # Display the selected model (disabled)
    st.text_input("LLM Model (Fixed for now)", value=gpt_mini_display, disabled=True)

    # Set the selected model and provider
    selected_model = gpt_mini_model
    selected_provider = gpt_mini_provider

    # Show reasoning option
    show_reasoning = st.checkbox("Show Analyst Reasoning", value=True, disabled=True)

    # Run button with warning if no analysts are selected
    if not selected_analysts:
        st.warning(
            "No team member selected and all technical analysts are deselected. The analysis will run with default settings (all analysts)."
        )

    run_button = st.button("Run Analysis", type="primary")

# Convert dates to string format for both tabs
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")

# Main content area
# Create tabs for main app sections
main_tabs = st.tabs(["Analysis", "Quick Charts"])

with main_tabs[0]:
    # Initialize session state for analysis results if not already set
    if 'analysis_run' not in st.session_state:
        st.session_state.analysis_run = False
        st.session_state.analysis_result = None
        st.session_state.analysis_tickers = []
        st.session_state.analysis_start_date = None
        st.session_state.analysis_end_date = None

    if run_button:
        # Initialize portfolio with cash amount and stock positions
        portfolio = {
            "cash": initial_cash,
            "margin_requirement": margin_requirement,
            "positions": {
                ticker: {
                    "long": 0,
                    "short": 0,
                    "long_cost_basis": 0.0,
                    "short_cost_basis": 0.0,
                }
                for ticker in tickers
            },
            "realized_gains": {
                ticker: {
                    "long": 0.0,
                    "short": 0.0,
                }
                for ticker in tickers
            },
        }

        # Show progress
        with st.spinner("Running InvestAI analysis..."):
            try:
                # Add a message about the analysis mode
                if not selected_analysts:
                    st.info("Running analysis with default settings (all analysts).")
                else:
                    # Create a message about which analysts are being used
                    analysis_components = []

                    if selected_team_member != "None":
                        analysis_components.append(
                            f"Team member: {selected_team_member}"
                        )

                    if selected_technicals:
                        tech_list = ", ".join(selected_technicals)
                        analysis_components.append(f"Technical analysts: {tech_list}")

                    analysis_message = "Running analysis with " + " and ".join(
                        analysis_components
                    )
                    st.info(analysis_message)

                # Run the hedge fund analysis
                result = run_hedge_fund(
                    tickers=tickers,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    portfolio=portfolio,
                    show_reasoning=show_reasoning,
                    selected_analysts=selected_analysts,  # This will be empty list if none selected
                    model_name=selected_model,
                    model_provider=selected_provider,
                )

                # Store results in session state
                if result and "decisions" in result:
                    st.session_state.analysis_run = True
                    st.session_state.analysis_result = result
                    st.session_state.analysis_tickers = tickers.copy()
                    st.session_state.analysis_start_date = start_date_str
                    st.session_state.analysis_end_date = end_date_str

                # Get results from session state or from the current run
                if st.session_state.analysis_run:
                    result = st.session_state.analysis_result
                    st.session_state.decisions = result["decisions"]
                    st.session_state.analyst_signals = result.get("analyst_signals", {})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)

    # Display analysis results if they exist in session state
    if st.session_state.analysis_run:
        # Create tabs for different views
        tabs = st.tabs(
            [
                "Portfolio Summary",
                "Detailed Analysis",
                "Analyst Signals",
            ]
        )

        # Portfolio Summary Tab
        with tabs[0]:
            st.markdown(
                '<div class="sub-header">Portfolio Summary</div>',
                unsafe_allow_html=True,
            )

            # Create a summary table
            summary_data = []
            for ticker, decision in st.session_state.decisions.items():
                action = decision.get("action", "").upper()
                quantity = decision.get("quantity", 0)
                confidence = decision.get("confidence", 0)

                # Style the action based on its type
                action_class = (
                    f"action-{action.lower()}"
                    if action.lower()
                    in ["buy", "sell", "hold", "short", "cover"]
                    else ""
                )
                styled_action = (
                    f'<span class="{action_class}">{action}</span>'
                )

                summary_data.append(
                    {
                        "Ticker": ticker,
                        "Action": styled_action,
                        "Quantity": quantity,
                        "Confidence": f"{confidence:.1f}%",
                    }
                )

            # Convert to DataFrame and display
            if summary_data:
                df = pd.DataFrame(summary_data)
                st.write(
                    df.to_html(escape=False, index=False),
                    unsafe_allow_html=True,
                )
            else:
                st.warning("No trading decisions available")

        # Detailed Analysis Tab
        with tabs[1]:
            st.markdown(
                '<div class="sub-header">Detailed Analysis</div>',
                unsafe_allow_html=True,
            )

            for ticker, decision in st.session_state.decisions.items():
                with st.expander(f"Analysis for {ticker}", expanded=True):
                    action = decision.get("action", "").upper()
                    quantity = decision.get("quantity", 0)
                    confidence = decision.get("confidence", 0)
                    reasoning = decision.get("reasoning", "")

                    # Create columns for layout
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(
                            f"**Action:** <span class='action-{action.lower()}'>{action}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**Quantity:** {quantity}")

                    with col2:
                        st.markdown(f"**Confidence:** {confidence:.1f}%")

                    st.markdown("**Reasoning:**")
                    st.markdown(reasoning)

        # Analyst Signals Tab
        with tabs[2]:
            st.markdown(
                '<div class="sub-header">Analyst Signals</div>',
                unsafe_allow_html=True,
            )

            for ticker in st.session_state.analysis_tickers:
                with st.expander(f"Signals for {ticker}", expanded=True):
                    signals_data = []

                    for agent, signals in st.session_state.analyst_signals.items():
                        if ticker not in signals:
                            continue

                        # Skip Risk Management agent in the signals section
                        if agent == "risk_management_agent":
                            continue

                        signal = signals[ticker]
                        agent_name = (
                            agent.replace("_agent", "")
                            .replace("_", " ")
                            .title()
                        )
                        signal_type = signal.get("signal", "").upper()
                        confidence = signal.get("confidence", 0)
                        reasoning = signal.get("reasoning", "")

                        # Style the signal based on its type
                        signal_class = ""
                        if signal_type == "BULLISH":
                            signal_class = "bullish"
                        elif signal_type == "BEARISH":
                            signal_class = "bearish"
                        elif signal_type == "NEUTRAL":
                            signal_class = "neutral"

                        styled_signal = f'<span class="{signal_class}">{signal_type}</span>'

                        md_text = format_reasoning_as_markdown(reasoning)
                        signals_data.append(
                            {
                                "Agent": agent_name,
                                "Signal": styled_signal,
                                "Confidence": f"{confidence}%",
                                "Reasoning": md_text  # Just store the markdown text, not st.markdown(md_text)
                            }
                        )


                    # Convert to DataFrame and display
                    if signals_data:
                        df = pd.DataFrame(signals_data)

                        # Count signal types for visualization
                        bullish_count = sum(
                            1
                            for item in signals_data
                            if "BULLISH" in item["Signal"]
                        )
                        bearish_count = sum(
                            1
                            for item in signals_data
                            if "BEARISH" in item["Signal"]
                        )
                        neutral_count = sum(
                            1
                            for item in signals_data
                            if "NEUTRAL" in item["Signal"]
                        )

                        # Create a pie chart of signals
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            fig = go.Figure(
                                data=[
                                    go.Pie(
                                        labels=[
                                            "Bullish",
                                            "Bearish",
                                            "Neutral",
                                        ],
                                        values=[
                                            bullish_count,
                                            bearish_count,
                                            neutral_count,
                                        ],
                                        marker_colors=[
                                            "#4CAF50",
                                            "#F44336",
                                            "#FFC107",
                                        ],
                                        hole=0.3,
                                    )
                                ]
                            )
                            fig.update_layout(
                                title_text=f"Signal Distribution for {ticker}"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            st.write(
                                df.to_html(escape=False, index=False),
                                unsafe_allow_html=True,
                            )
                    else:
                        st.warning(
                            f"No analyst signals available for {ticker}"
                        )

    # Add Technical Charts section if analysis has been run successfully
    if st.session_state.analysis_run:
        st.markdown("---")
        st.markdown('<div class="sub-header">Technical Charts</div>', unsafe_allow_html=True)

        # Add indicator selection
        st.markdown("### Chart Settings")
        col1, col2 = st.columns([1, 2])

        with col1:
            # Get the list of tickers from the stored analysis
            analysis_tickers = st.session_state.analysis_tickers

            # Simple ticker selection
            selected_ticker = st.selectbox(
                "Select Ticker",
                analysis_tickers,
                key="analysis_chart_ticker"
            )

            # Technical indicator selection
            st.markdown("### Technical Indicators")
            indicators = []

            # Use unique keys for each checkbox
            if st.checkbox("Moving Averages", value=True, key="analysis_ma"):
                indicators.extend(["SMA20", "SMA50", "SMA200"])

            if st.checkbox("Bollinger Bands", value=True, key="analysis_bb"):
                indicators.append("BB")

            if st.checkbox("MACD", value=True, key="analysis_macd"):
                indicators.append("MACD")

            if st.checkbox("RSI", value=True, key="analysis_rsi"):
                indicators.append("RSI")

            if st.checkbox("Volume", value=True, key="analysis_volume"):
                indicators.append("Volume")

            # Add a button to generate chart
            generate_chart = st.button("Generate Chart", key="analysis_generate_chart")

        with col2:
            # Always generate the chart when the button is pressed or when the form is first loaded
            if generate_chart or 'chart_first_load' not in st.session_state:
                # Mark that we've loaded the chart at least once
                st.session_state.chart_first_load = True

                with st.spinner(f"Loading chart for {selected_ticker}..."):
                    try:
                        # Generate a new chart using the stored dates
                        print(f"Generating chart for {selected_ticker} with indicators: {indicators}")
                        fig = get_stock_chart_with_indicators(
                            selected_ticker,
                            st.session_state.analysis_start_date,
                            st.session_state.analysis_end_date,
                            indicators,
                        )
                        # Store the chart in session state
                        if 'charts' not in st.session_state:
                            st.session_state.charts = {}
                        # Use a key that includes the ticker and indicators to cache different chart configurations
                        chart_key = f"{selected_ticker}_{'-'.join(indicators)}"
                        st.session_state.charts[chart_key] = fig
                    except Exception as e:
                        st.error(f"Error generating chart: {str(e)}")
                        st.exception(e)

            # Display the chart if it exists in session state
            if 'charts' in st.session_state:
                chart_key = f"{selected_ticker}_{'-'.join(indicators)}"
                if chart_key in st.session_state.charts:
                    st.plotly_chart(st.session_state.charts[chart_key], use_container_width=True)

                    # Add a note about the chart
                    st.info(
                       """
                     This interactive chart shows historical price data with selected technical indicators.
                    You can zoom, pan, and hover over data points for more information.
                   """
                     )
                else:
                    # If we don't have this specific chart configuration, generate it
                    with st.spinner(f"Loading chart for {selected_ticker}..."):
                        try:
                            fig = get_stock_chart_with_indicators(
                                selected_ticker,
                                st.session_state.analysis_start_date,
                                st.session_state.analysis_end_date,
                                indicators,
                            )
                            st.session_state.charts[chart_key] = fig
                            st.plotly_chart(fig, use_container_width=True)

                            # Add a note about the chart
                            st.info(
                               """
                             This interactive chart shows historical price data with selected technical indicators.
                            You can zoom, pan, and hover over data points for more information.
                           """
                             )
                        except Exception as e:
                            st.error(f"Error generating chart: {str(e)}")
                            st.exception(e)

# Quick Charts tab
with main_tabs[1]:
    st.markdown(
        '<div class="sub-header">Quick Technical Charts</div>', unsafe_allow_html=True
    )
    st.markdown("View technical charts for any stock without running a full analysis")

    # Input for ticker symbol
    quick_ticker = st.text_input("Enter Stock Ticker", "AAPL")

    # Date range selection
    st.markdown("### Date Range")
    quick_end_date = st.date_input("End Date (Quick Chart)", datetime.now())
    quick_date_range_options = {
        "1 Month": quick_end_date - relativedelta(months=1),
        "3 Months": quick_end_date - relativedelta(months=3),
        "6 Months": quick_end_date - relativedelta(months=6),
        "1 Year": quick_end_date - relativedelta(years=1),
        "5 Years": quick_end_date - relativedelta(years=5),
        "Custom": "custom",
    }
    quick_date_range = st.selectbox(
        "Select Date Range (Quick Chart)", list(quick_date_range_options.keys())
    )

    if quick_date_range == "Custom":
        quick_start_date = st.date_input(
            "Start Date (Quick Chart)", quick_end_date - relativedelta(months=3)
        )
    else:
        quick_start_date = quick_date_range_options[quick_date_range]

    # Convert dates to string format
    quick_start_date_str = quick_start_date.strftime("%Y-%m-%d")
    quick_end_date_str = quick_end_date.strftime("%Y-%m-%d")

    # Technical indicator selection
    st.markdown("### Technical Indicators")
    col1, col2, col3 = st.columns(3)

    quick_indicators = []

    with col1:
        if st.checkbox("Moving Averages (Quick)", value=True, key="quick_ma"):
            quick_indicators.extend(["SMA20", "SMA50", "SMA200"])

        if st.checkbox("Bollinger Bands (Quick)", value=True, key="quick_bb"):
            quick_indicators.append("BB")

    with col2:
        if st.checkbox("MACD (Quick)", value=True, key="quick_macd"):
            quick_indicators.append("MACD")

        if st.checkbox("RSI (Quick)", value=True, key="quick_rsi"):
            quick_indicators.append("RSI")

    with col3:
        if st.checkbox("Volume (Quick)", value=True, key="quick_volume"):
            quick_indicators.append("Volume")

    # Button to generate chart
    generate_quick_chart = st.button("Generate Chart", type="primary", key="quick_chart_button")

    # Display the chart if the button is pressed
    if generate_quick_chart:
        with st.spinner(f"Loading chart for {quick_ticker}..."):
            try:
                print(f"Generating quick chart for {quick_ticker} with indicators: {quick_indicators}")
                fig = get_stock_chart_with_indicators(
                    quick_ticker,
                    quick_start_date_str,
                    quick_end_date_str,
                    quick_indicators,
                )
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)

                # Add a note about the chart
                st.info(
                    """
                This interactive chart shows historical price data with selected technical indicators.
                You can zoom, pan, and hover over data points for more information.
                """
                )
            except Exception as e:
                st.error(f"Error generating chart: {str(e)}")
                st.exception(e)

# Add information about the application
with st.expander("About this Application"):
    st.markdown(
        """
    ### InvestAI Analyst Platform
    
    This application uses AI analysts to analyze stocks and make trading decisions. Each analyst has a different 
    investment philosophy and approach to the market. The application combines these signals to make final trading decisions.
    
    #### How it works:
    1. Select the stocks you want to analyze
    2. Choose the date range for historical data
    3. Configure your portfolio settings
    4. Optionally select a single team member to focus the analysis
    5. Select which technical analysts to include (all are selected by default)
    6. Run the analysis to get trading recommendations

    #### Technical Charts:
    The platform includes interactive technical charts with various indicators:
    - Moving Averages (SMA20, SMA50, SMA200)
    - Bollinger Bands
    - MACD (Moving Average Convergence Divergence)
    - RSI (Relative Strength Index)
    - Volume Analysis

    You can access these charts in two ways:
    - In the Analysis results under the "Technical Charts" tab
    - Directly through the "Quick Charts" tab without running a full analysis

    #### Analysis Options:
    - **AI Team**: Famous investors like Warren Buffett, Charlie Munger, etc. with distinct investment philosophies
       (you can select one or none)
    - **AI Technicals**: Specialized analysts focusing on technical, fundamental, sentiment, and valuation aspects
       (all are selected by default, you can deselect any you don't want to include)
    
    The final decision is made by combining all analyst signals and applying risk management rules.
    """
    )

if __name__ == "__main__":
    # This will only run when the script is executed directly
    pass
