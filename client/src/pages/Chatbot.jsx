import React, { useEffect, useRef, useState } from "react";
import { IoMdAdd } from "react-icons/io";
import { IoSend } from "react-icons/io5";
import { Link } from "react-router-dom";
import { FiEdit2 } from "react-icons/fi";
import { MdDashboard } from "react-icons/md";
import { FaReceipt } from "react-icons/fa";
import { getAuth } from "firebase/auth";
import {
  Bar,
  Line,
  Pie
} from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Tooltip,
  Legend
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Tooltip,
  Legend
);

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;

const Chatbot = () => {
  const [chats, setChats] = useState([
    {
      id: 1,
      title: "New Chat",
      messages: [{ sender: "bot", text: "Hi! How can I help you today?" }],
    },
  ]);
  const [activeChatIndex, setActiveChatIndex] = useState(0);
  const [inputText, setInputText] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const inputRef = useRef(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const renderChart = (visualization) => {
    const { type, fields, caption } = visualization || {};
    if (!fields || !fields.x_axis || !fields.y_axis || !fields.x_axis.length || !fields.y_axis.length) return null;

    const baseDataset = {
      label: caption,
      data: fields.y_axis,
      borderColor: "#2563eb",
      backgroundColor: "#3b82f6",
      borderWidth: 2,
    };

    const pieColors = [
      "#4285F4", "#EA4335", "#FBBC05", "#34A853",
      "#A142F4", "#00ACC1", "#F4511E", "#C0CA33"
    ];

    const pieDataset = {
      label: caption,
      data: fields.y_axis,
      backgroundColor: pieColors.slice(0, fields.y_axis.length),
      borderWidth: 1,
    };

    const options = {
      responsive: true,
      maintainAspectRatio: type === "pie_chart" ? false : true,
      plugins: {
        legend: {
          display: type === "pie_chart",
          position: "bottom",
        },
      },
    };

    switch (type) {
      case "bar_chart":
        return (
          <div className="h-[210px] mx-auto">
            <Bar data={{ labels: fields.x_axis, datasets: [baseDataset] }} options={options} />
          </div>
        )
      case "line_chart":
        return (
          <div className="h-[210px] mx-auto">
            <Line data={{ labels: fields.x_axis, datasets: [baseDataset] }} options={options} />
          </div>
        )
      case "pie_chart":
        return (
          <div className="h-[210px] mx-auto">
            <Pie data={{ labels: fields.x_axis, datasets: [pieDataset] }} options={options} />
          </div>
        );
      default:
        return null;
    }
  };

  const handleSendMessage = async (text) => {
    if (!text.trim()) return;

    const updatedChats = [...chats];
    updatedChats[activeChatIndex].messages.push({ sender: "user", text });
    updatedChats[activeChatIndex].messages.push({ sender: "bot", text: "RASEED is thinking…" });
    setChats(updatedChats);
    setInputText("");
    setIsThinking(true);

    try {
      const auth = getAuth();
      const user = auth.currentUser;
      if (!user) throw new Error("User not authenticated");

      const token = await user.getIdToken();

      const response = await fetch(`${BACKEND_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ query: text }),
      });

      const data = await response.json();
      updatedChats[activeChatIndex].messages.pop(); // remove thinking msg

      const { insights, visualization, explanation } = data.reply || {};
      updatedChats[activeChatIndex].messages.push({
        sender: "bot",
        text: insights,
        visualization,
        explanation,
      });
    } catch (e) {
      console.error("Error sending message:", e);
      updatedChats[activeChatIndex].messages.pop();
      updatedChats[activeChatIndex].messages.push({
        sender: "bot",
        text: "Error getting response from RASEED.",
      });
    } finally {
      setChats([...updatedChats]);
      setIsThinking(false);
    }
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chats]);

  const handleNewChat = () => {
    const newChat = {
      id: Date.now(),
      title: "New Chat",
      messages: [{ sender: "bot", text: "Hi! How can I help you today?" }],
    };
    setChats([newChat, ...chats]);
    setActiveChatIndex(0);
    setInputText("");
    setTimeout(() => inputRef.current?.focus(), 50);
  };

  useEffect(() => {
    if (!isThinking) {
      inputRef.current?.focus();
    }
  }, [activeChatIndex, isThinking]);

  const quickQuestions = [
    { label: "Summarize My Receipts", bg: "bg-blue-500" },
    { label: "What Did I Spend The Most On?", bg: "bg-red-500" },
    { label: "Any Unusual Expenses?", bg: "bg-green-500" },
    { label: "How Can I Save More?", bg: "bg-yellow-500" },
  ];

  return (
    <div className="flex h-full bg-white rounded-xl shadow-lg overflow-hidden">
      {/* Sidebar */}
      <div className="w-1/4 bg-gray-100 p-4 border-r border-gray-200 flex flex-col">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-gray-800">Chats</h2>
          <button
            onClick={handleNewChat}
            className="p-2 bg-blue-500 text-white rounded-full hover:bg-blue-600"
          >
            <IoMdAdd />
          </button>
        </div>

        {/* Chat list with rename buttons */}
        <div className="flex-1 overflow-y-auto space-y-2">
          {chats.map((chat, index) => (
            <div
              key={chat.id}
              className={`group flex items-center justify-between p-3 rounded-lg cursor-pointer ${
                index === activeChatIndex
                  ? "bg-blue-100 text-blue-800 font-semibold"
                  : "bg-white text-gray-800 hover:bg-gray-200"
              }`}
            >
              <div
                onClick={() => setActiveChatIndex(index)}
                className="flex-1 truncate"
              >
                {chat.title}
              </div>
              <button
                onClick={() => {
                  const newTitle = prompt("Rename chat:", chat.title);
                  if (newTitle) {
                    const updatedChats = [...chats];
                    updatedChats[index].title = newTitle;
                    setChats(updatedChats);
                  }
                }}
                className="ml-2 text-gray-500 hover:text-gray-700"
              >
                <FiEdit2 size={16} />
              </button>
            </div>
          ))}
        </div>

        {/* Footer Quick Links */}
        <hr className="my-4 border-gray-300" />
        <h4 className="text-sm font-medium text-gray-500 mb-2">Quick Links</h4>
        <div className="space-y-3">
          <Link
            to="/dashboard"
            className="flex items-center space-x-2 text-gray-700 hover:text-blue-600"
          >
            <MdDashboard size={20} />
            <span>Dashboard</span>
          </Link>
          <Link
            to="/history"
            className="flex items-center space-x-2 text-gray-700 hover:text-blue-600"
          >
            <FaReceipt size={18} />
            <span>My Receipts</span>
          </Link>
        </div>
      </div>

      {/* Chat Area */}
      <div className="w-3/4 flex flex-col">
        <div className="ml-3 p-4 border-b border-gray-200 bg-white flex items-center justify-between">
          <h3 className="text-md font-semibold text-gray-800">
            {chats[activeChatIndex]?.title || "Chat"}
          </h3>
          <div className="relative">
            <button
              onClick={() => setDropdownOpen(!dropdownOpen)}
              className="px-4 py-2 text-sm bg-gray-100 border rounded-md hover:bg-gray-200"
            >
              Quick Questions
            </button>
            {dropdownOpen && (
              <div className="absolute right-0 mt-2 w-64 bg-white shadow-lg border rounded-md z-10">
                {quickQuestions.map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      handleSendMessage(q.label);
                      setDropdownOpen(false);
                    }}
                    className={`block w-full text-left px-4 py-2 ease-in-out rounded-md hover:text-white hover:${q.bg}`}
                  >
                    {q.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6 hide-scrollbar">
          {chats[activeChatIndex]?.messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`px-4 py-2 rounded-lg max-w-xl w-max ${msg.sender === "user" ? "bg-blue-500 text-white" : "bg-gray-100 text-gray-800"}`}>
                {typeof msg.text === "string" ? <p>{msg.text}</p> : null}
                {msg.visualization && renderChart(msg.visualization)}
                {msg.visualization?.caption && <p className="mt-2 text-sm text-gray-600 font-medium">{msg.visualization.caption}</p>}
                {msg.explanation && <p className="mt-1 text-xs text-gray-500 italic">{msg.explanation}</p>}
              </div>
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        {/* Input */}
        <div className="p-4 border-t border-gray-200 bg-white">
          <form
            className="flex items-center space-x-4"
            onSubmit={(e) => {
              e.preventDefault();
              handleSendMessage(inputText);
            }}
          >
            <input
              ref={inputRef}
              type="text"
              className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder={
                isThinking ? "RASEED is thinking..." : "Type your message..."
              }
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              disabled={isThinking}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage(inputText);
                }
              }}
            />
            <button
              type="submit"
              disabled={isThinking}
              className="p-2 bg-blue-500 text-white rounded-full hover:bg-blue-600"
            >
              <IoSend size={20} />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;
