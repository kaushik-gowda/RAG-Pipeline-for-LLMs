# ============================================================
# main.py – CLI interface for the RAG Pipeline
# ============================================================
"""
Run this script to interact with the RAG pipeline from the
command line.

Usage:
    python main.py
"""

from rag_pipeline import RAGPipeline


def main() -> None:
    print("=" * 60)
    print("  RAG Pipeline for LLMs – Command Line Interface")
    print("=" * 60)

    pipeline = RAGPipeline()

    # --- Step 1: Ingest a topic ---
    topic = input("\n📚 Enter a topic to learn about: ").strip()
    if not topic:
        print("No topic provided. Exiting.")
        return

    result = pipeline.ingest(topic)
    if not result["success"]:
        print(f"\n❌ {result['message']}")
        return

    print(f"\n✅ {result['message']}")

    # --- Step 2: Q&A loop ---
    print("\nYou can now ask questions about this topic.")
    print("Type 'quit' to exit, 'topic' to change the topic.\n")

    while True:
        question = input("❓ Your question: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break

        if question.lower() == "topic":
            topic = input("\n📚 Enter a new topic: ").strip()
            if topic:
                result = pipeline.ingest(topic)
                if result["success"]:
                    print(f"\n✅ {result['message']}\n")
                else:
                    print(f"\n❌ {result['message']}\n")
            continue

        if not question:
            continue

        answer_data = pipeline.ask(question)

        if "error" in answer_data:
            print(f"\n❌ {answer_data['error']}\n")
            continue

        print(f"\n💡 Answer: {answer_data['answer']}")
        print(f"   Confidence: {answer_data['score']:.2%}")
        print(f"   Based on {len(answer_data['retrieved_chunks'])} retrieved chunks.\n")


if __name__ == "__main__":
    main()
