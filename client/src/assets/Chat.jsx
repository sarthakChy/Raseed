// import { useEffect, useRef, useState } from "react";
// import ChatHeader from "./ChatHeader";
// import MessageList from "./MessageList";
// import MessageInput from "./MessageInput";
// import { useAuth } from "../context/AuthContext";

// export default function Chat() {
//   const CHAT_SERVICE_URL = import.meta.env.VITE_CHAT_SERVICE_URL;
//   const { user } = useAuth();
//   const uid = null;
//   const username = user?.user_metadata?.name || "You";

//   const [messages, setMessages] = useState([]);
//   const [isThinking, setIsThinking] = useState(false);
//   const messageEndRef = useRef(null);

//   useEffect(() => {
//     messageEndRef.current?.scrollIntoView({ behavior: "smooth" });
//   }, [messages]);

//   const handleSend = async (text) => {
//     if (!text) return;

//     const userMessage = {
//       mid: crypto.randomUUID(),
//       uid,
//       sender: username,
//       text,
//       time: new Date().toISOString(),
//     };

//     setMessages((prev) => [...prev, userMessage]);
//     setIsThinking(true);

//     try {
//       const token = await user?.getIdToken?.();
//       if (!token) throw new Error("No auth token found");

//       const res = await fetch(`${CHAT_SERVICE_URL}/chat`, {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//           Authorization: `Bearer ${token}`,
//         },
//         body: JSON.stringify({ uid, message: text }),
//       });

//       if (!res.ok) throw new Error("AI response failed");

//       const data = await res.json();
//       console.log(data);
      

//       const aiMessage = {
//         mid: crypto.randomUUID(),
//         uid: "ai",
//         sender: "RASEED",
//         text: data.reply,
//         time: new Date().toISOString(),
//       };

//       setMessages((prev) =>
//         [...prev.filter((m) => m.mid !== "typing-indicator"), aiMessage]
//       );
//     } catch (err) {
//       const errMessage = {
//         mid: crypto.randomUUID(),
//         uid: "ai",
//         sender: "RASEED",
//         text: "Oops! Something went wrong.",
//         time: new Date().toISOString(),
//       };

//       setMessages((prev) =>
//         [...prev.filter((m) => m.mid !== "typing-indicator"), errMessage]
//       );
//     } finally {
//       setIsThinking(false);
//     }
//   };

//   return (
//     <div className="flex flex-col h-screen bg-white text-slate-800">
//       <ChatHeader isThinking={isThinking} />
//       <div className="flex-1 bg-blue-200 overflow-y-auto p-4 hide-scrollbar">
//         <MessageList messages={messages} currentUserId={uid} />
//         <div ref={messageEndRef} />
//       </div>
//       <div className="p-4 bg-blue-500">
//         <MessageInput onSend={handleSend} isThinking={isThinking} />
//       </div>
//     </div>
//   );
// }
