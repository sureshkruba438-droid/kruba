# ============================================================
# AI STUDENT SUPPORT ASSISTANT - RAG + TOOLS + MEMORY (Compact)
# ============================================================

import re, json
from datetime import datetime
from collections import Counter

PROJECT_NAME = "AI Student Support Assistant"
VERSION = "1.0"

KNOWLEDGE_BASE = [
    {"id": "REG001", "category": "Regulations", "title": "Attendance Regulations",
     "content": "Students should maintain the minimum attendance required by the applicable "
                "college and university regulations. Attendance eligibility may be checked "
                "before semester examinations. Monitor attendance via the academic portal."},
    {"id": "REG002", "category": "Regulations", "title": "Examination Eligibility",
     "content": "Students must satisfy academic and attendance requirements before appearing "
                "for semester examinations. Complete exam applications and fees before the deadline."},
    {"id": "SYL001", "category": "Syllabus", "title": "AI and Data Science Syllabus",
     "content": "The AI and Data Science curriculum includes programming, data structures, "
                "databases, statistics, machine learning, deep learning, big data analytics, "
                "cloud computing, data visualization and project work."},
    {"id": "SYL002", "category": "Syllabus", "title": "Data Analytics",
     "content": "Key data analytics areas include data cleaning, exploratory data analysis, "
                "descriptive statistics, visualization, correlation and reporting."},
    {"id": "FAQ001", "category": "FAQ", "title": "Library Information",
     "content": "The library offers textbooks, reference materials, journals and digital "
                "resources. Check borrowing limits and hours from the latest library notice."},
    {"id": "FAQ002", "category": "FAQ", "title": "Academic Project",
     "content": "An academic project includes problem identification, requirement analysis, "
                "design, implementation, testing, documentation and final presentation."},
    {"id": "PLAC001", "category": "Placement", "title": "Placement Preparation",
     "content": "Placement preparation includes aptitude practice, communication skills, "
                "resume preparation, technical fundamentals and interview practice."},
    {"id": "STUDY001", "category": "Study", "title": "Study Planning",
     "content": "A good study plan divides subjects into topics, sets weekly goals, reserves "
                "time for revision and includes practice questions."},
    {"id": "NOTICE001", "category": "Notices", "title": "Academic Notices",
     "content": "Notices cover exams, assignments, workshops, project reviews, holidays and "
                "registrations. Verify date, time and venue from the official notice."},
]

STOP_WORDS = {"the", "is", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at",
              "what", "how", "when", "where", "why", "can", "i", "my", "me", "about",
              "tell", "give", "from", "with", "college", "student", "students",
              "please", "do", "does"}

TOPIC_KEYWORDS = {
    "Regulations": {"attendance", "regulation", "rules", "eligibility", "exam",
                     "examination", "university", "minimum"},
    "Syllabus": {"syllabus", "subject", "subjects", "unit", "units", "curriculum",
                 "machine", "learning", "analytics", "data", "ai"},
    "FAQ": {"library", "book", "borrow", "return", "project", "guide", "department"},
    "Placement": {"placement", "job", "resume", "interview", "aptitude", "company", "career"},
    "Study": {"study", "schedule", "revision", "plan", "prepare", "preparation"},
    "Notices": {"notice", "deadline", "date", "venue", "workshop", "registration",
                "assignment", "review", "holiday"},
}


def clean_text(text):
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return [w for w in clean_text(text).split() if w not in STOP_WORDS]


def detect_topic(question):
    words = set(tokenize(question))
    scores = {t: len(words & kw) for t, kw in TOPIC_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "General", 0.0
    return best, round(min(1.0, scores[best] / 3), 2)


class RAGEngine:
    """Simple keyword-overlap retrieval engine (title matches weighted higher)."""

    def __init__(self, documents):
        self.documents = documents
        self.index = {d["id"]: set(tokenize(d["title"] + " " + d["content"])) for d in documents}

    def retrieve(self, query, top_k=3):
        q = set(tokenize(query))
        scored = []
        for d in self.documents:
            score = len(q & self.index[d["id"]]) + 2 * len(q & set(tokenize(d["title"])))
            if score > 0:
                scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"score": s, **{k: d[k] for k in ("id", "category", "title", "content")}}
                for s, d in scored[:top_k]]


class ConversationMemory:
    def __init__(self, max_memory=20):
        self.max_memory = max_memory
        self.memory = []

    def save(self, question, topic, answer):
        self.memory.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question, "topic": topic, "answer": answer[:300],
        })
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def get_recent(self, count=5):
        return self.memory[-count:]

    def clear(self):
        self.memory.clear()


class AssistantTools:
    def __init__(self, rag_engine):
        self.rag = rag_engine

    def search_tool(self, query):
        results = self.rag.retrieve(query)
        if not results:
            return "No matching information was found."
        lines = ["Search Results:"]
        lines += [f"{i}. {r['title']} ({r['category']}) - Score: {r['score']}"
                  for i, r in enumerate(results, 1)]
        return "\n".join(lines)

    def topic_tool(self, question):
        topic, confidence = detect_topic(question)
        return {"topic": topic, "confidence": confidence}

    QUIZ_BANK = {
        "Regulations": [("Why is attendance important?",
                          "It helps students satisfy attendance eligibility."),
                         ("Where should students check exam info?",
                          "The latest official examination notice.")],
        "Syllabus": [("Name one AI and Data Science subject.", "Machine Learning."),
                     ("What is data visualization?", "Presenting data visually.")],
        "Placement": [("Name one placement preparation activity.", "Aptitude practice."),
                       ("Why is resume preparation important?",
                        "It presents the student's skills and experience.")],
        "Study": [("Why are weekly study goals useful?",
                   "They make study progress easier to track.")],
    }

    def quiz_tool(self, topic):
        questions = self.QUIZ_BANK.get(topic, self.QUIZ_BANK["Study"])
        lines = ["\n===== MINI QUIZ ====="]
        for i, (q, a) in enumerate(questions, 1):
            lines += [f"\nQ{i}. {q}", f"Answer: {a}"]
        return "\n".join(lines)


class ResponseGenerator:
    @staticmethod
    def generate(topic, confidence, results):
        if not results:
            return ("\nI could not find a matching document in the current college knowledge "
                    "base.\nPlease check the latest official college, department or university "
                    "notice for current information.\n")
        best = results[0]
        lines = [f"Topic: {topic}", f"Confidence: {confidence}", f"Source: {best['title']}",
                 "", best["content"]]
        if len(results) > 1:
            lines.append("\nRelated Information:")
            lines += [f"- {r['title']}" for r in results[1:]]
        lines.append("\nNote: Sample knowledge base — verify official information for real decisions.")
        return "\n".join(lines)


class StudentSupportAgent:
    def __init__(self):
        self.rag = RAGEngine(KNOWLEDGE_BASE)
        self.memory = ConversationMemory()
        self.tools = AssistantTools(self.rag)
        self.question_count = 0
        self.tool_count = 0
        self.start_time = datetime.now()

    def answer(self, question):
        self.question_count += 1
        topic, confidence = detect_topic(question)
        results = self.rag.retrieve(question)
        self.tool_count += 1
        self.tools.search_tool(question)  # tool call (kept for logging/telemetry)
        answer = ResponseGenerator.generate(topic, confidence, results)
        self.memory.save(question, topic, answer)
        return answer

    def show_memory(self):
        recent = self.memory.get_recent()
        if not recent:
            return "No conversation memory available."
        lines = ["\n===== CONVERSATION MEMORY ====="]
        for item in recent:
            lines += [f"\nTime: {item['time']}", f"Topic: {item['topic']}",
                      f"Question: {item['question']}"]
        return "\n".join(lines)

    def report(self):
        duration = datetime.now() - self.start_time
        topics = Counter(item["topic"] for item in self.memory.memory)
        lines = ["\n================================", "       SESSION REPORT",
                  "================================", f"Project: {PROJECT_NAME}",
                  f"Questions: {self.question_count}", f"Tool Calls: {self.tool_count}",
                  f"Memory Items: {len(self.memory.memory)}", f"Session Duration: {duration}",
                  "\nTopics:"]
        lines += [f"- {t}: {c}" for t, c in topics.most_common()] if topics else ["- No questions yet"]
        return "\n".join(lines)

    def export_session(self, filename="student_support_session.json"):
        data = {"project": PROJECT_NAME, "version": VERSION,
                "generated_at": datetime.now().isoformat(),
                "questions_answered": self.question_count,
                "tool_calls": self.tool_count, "memory": self.memory.memory}
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return f"Session exported to {filename}"


HELP_TEXT = """
Ask about: attendance, syllabus, exams, placements, study plans, projects, notices, library.

Commands: help | topics | quiz | memory | report | clear | export | exit
"""

TOPICS_TEXT = "Topics: Regulations, Syllabus, FAQ, Placement, Study, Notices"


def run_demo(agent):
    demo_questions = [
        "What are the attendance regulations?",
        "Tell me about the AI and Data Science syllabus.",
        "How can I prepare for placements?",
        "How should I plan my studies?",
    ]
    print("\n================ DEMO MODE ================\n")
    for q in demo_questions:
        print(f"Student: {q}\n\nAssistant:\n{agent.answer(q)}")
        print("\n--------------------------------------------\n")


def main():
    agent = StudentSupportAgent()
    print("============================================================\n"
          "             AI STUDENT SUPPORT ASSISTANT\n"
          "============================================================\n"
          "Capabilities: RAG + Tools + Memory\n"
          "Type 'help' for commands, 'exit' to quit.\n")

    while True:
        try:
            user_input = input("Student > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nApplication closed.")
            break
        if not user_input:
            continue
        cmd = user_input.lower()

        if cmd in ("exit", "quit", "bye"):
            print("\nThank you for using AI Student Support Assistant.")
            print(agent.report())
            print("\n" + agent.export_session())
            break
        elif cmd == "help":
            print(HELP_TEXT)
        elif cmd == "topics":
            print(TOPICS_TEXT)
        elif cmd == "memory":
            print(agent.show_memory())
        elif cmd == "report":
            print(agent.report())
        elif cmd == "quiz":
            topic, _ = detect_topic("study preparation")
            print(agent.tools.quiz_tool(topic))
        elif cmd == "clear":
            agent.memory.clear()
            print("Conversation memory cleared.")
        elif cmd == "export":
            print(agent.export_session())
        else:
            print("\nAssistant:")
            print(agent.answer(user_input))
            print()


if __name__ == "__main__":
    main()
