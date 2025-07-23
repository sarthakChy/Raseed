// import { useRef, useState, useEffect } from "react";
// import { SendHorizonal } from "lucide-react";

// export default function MessageInput({ onSend, isThinking }) {
//   const [text, setText] = useState("");
//   const textareaRef = useRef(null);

//   // Refocus the textarea when AI finishes responding
//   useEffect(() => {
//     if (!isThinking && textareaRef.current) {
//       textareaRef.current.focus();
//     }
//   }, [isThinking]);

//   const handleSend = () => {
//     const trimmed = text.trim();
//     if (!trimmed) return;

//     onSend(trimmed);
//     setText("");
//     resizeTextarea(""); // Reset height
//   };

//   const handleChange = (e) => {
//     const value = e.target.value;
//     setText(value);
//     resizeTextarea(value);
//   };

//   const handleKeyDown = (e) => {
//     if (e.key === "Enter" && !e.shiftKey) {
//       e.preventDefault();
//       handleSend();
//     }
//   };

//   const resizeTextarea = (value) => {
//     const el = textareaRef.current;
//     if (!el) return;
//     el.style.height = "auto";
//     el.style.height = Math.min(el.scrollHeight, 80) + "px"; // Max 5 lines (5 * 16px line height)
//   };

//   return (
//     <div className="flex items-center gap-3 border-blue-200">
//       <textarea
//         ref={textareaRef}
//         value={text}
//         onChange={handleChange}
//         onKeyDown={handleKeyDown}
//         placeholder={isThinking ? "RASEED is thinking…" : "Ask something..."}
//         disabled={isThinking}
//         className="flex-1 resize-none bg-white hide-scrollbar rounded-lg p-3 text-sm text-blue-900 placeholder:text-blue-400 focus:outline-none max-h-[7.5rem] overflow-y-auto border border-blue-200"
//         style={{ lineHeight: "1.5rem" }}
//       />

//       <button
//         onClick={handleSend}
//         disabled={isThinking}
//         className="text-blue-600 disabled:opacity-40"
//       >
//         <SendHorizonal size={30} className="text-black hover:text-white ease-in-out" />
//       </button>
//     </div>
//   );
// }
