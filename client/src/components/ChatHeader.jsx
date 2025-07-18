export default function ChatHeader({ isThinking }) {
  return (
    <header className="flex items-center justify-between px-6 py-4 border-b border-blue-200 bg-blue-500">
      <div className="flex flex-col">
        <h2 className="text-2xl font-semibold text-black">RASEED</h2>
        <p className="text-xs text-black">
          {isThinking ? "RASEED is thinking…" : "Ask me anything about your receipts"}
        </p>
      </div>
    </header>
  );
}
