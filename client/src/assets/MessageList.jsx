// import { useRef, useEffect } from "react";
// import ReactMarkdown from "react-markdown";
// import rehypeRaw from "rehype-raw";

// export default function MessageList({ messages = [], currentUserId }) {
//   const msgRefs = useRef({});

//   useEffect(() => {
//     const lastMsg = messages[messages.length - 1];
//     if (lastMsg) {
//       const el = msgRefs.current[lastMsg.mid];
//       el?.scrollIntoView({ behavior: "smooth", block: "end" });
//     }
//   }, [messages]);

//   return (
//     <div className="flex flex-col gap-3">
//       {messages.map((msg) => {
//         const isOwn = msg.uid === currentUserId;
//         const isTyping = msg.mid === "typing-indicator";

//         return (
//           <div
//             key={msg.mid}
//             ref={(el) => (msgRefs.current[msg.mid] = el)}
//             data-mid={msg.mid}
//             className={`max-w-[75%] px-4 py-2 rounded-lg shadow-sm text-sm break-words transition-colors ${
//               isOwn
//                 ? "self-end bg-blue-600 text-white"
//                 : isTyping
//                 ? "self-start bg-blue-100 text-blue-800 italic"
//                 : "self-start bg-gray-100 text-blue-900"
//             }`}
//           >
//             {!isOwn && !isTyping && (
//               <div className="text-xs font-semibold mb-1 text-blue-700">
//                 {msg.sender}
//               </div>
//             )}

//             <div className="whitespace-pre-wrap max-w-none text-sm leading-relaxed">
//               {isTyping ? (
//                 "RASEED is thinking…"
//               ) : (
//                 <ReactMarkdown
//                   rehypePlugins={[rehypeRaw]}
//                   components={{
//   strong: ({ children }) => <strong className="font-bold">{children}</strong>,
//   em: ({ children }) => <em className="italic">{children}</em>,
//   u: ({ children }) => <u className="underline">{children}</u>,
//   p: ({ children }) => <p className="mb-2">{children}</p>,
//   ul: ({ children }) => <ul className="list-disc list-outside ml-5 mb-2">{children}</ul>,
//   ol: ({ children }) => <ol className="list-decimal list-outside ml-5 mb-2">{children}</ol>,
//   li: ({ children }) => <li className="pl-1">{children}</li>,
//   code: ({ children }) => (
//     <code className="bg-gray-900 text-white px-1 py-0.5 rounded text-sm">{children}</code>
//   ),
//   pre: ({ children }) => (
//     <pre className="bg-gray-900 text-white p-4 rounded-md overflow-x-auto text-sm">
//       {children}
//     </pre>
//   ),
// }}

//                 >
//                   {msg.text}
//                 </ReactMarkdown>
//               )}
//             </div>
//           </div>
//         );
//       })}
//     </div>
//   );
// }
