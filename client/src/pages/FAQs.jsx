import React, { useState } from "react";
import { BsChevronDown } from "react-icons/bs";
import Header from "../components/Header"; // ✅ Added

const faqData = [
  {
    question: "What is your refund policy?",
    answer:
      "Our refund policy allows for returns within 30 days of purchase. Please visit our returns page for detailed instructions and to initiate a refund request.",
    color: "bg-red-500",
  },
  {
    question: "How do I cancel my subscription?",
    answer:
      "You can cancel your subscription at any time through your account settings. Once canceled, you will retain access until the end of your current billing period.",
    color: "bg-blue-500",
  },
  {
    question: "Is there a discount for annual billing?",
    answer:
      "Yes, we offer a significant discount for users who choose annual billing. You can save up to 20% compared to paying monthly.",
    color: "bg-yellow-400",
  },
  {
    question: "How can I contact support?",
    answer:
      "Our support team is available 24/7. You can reach us via email at support@raseed.com, or through the live chat feature on our website.",
    color: "bg-green-500",
  },
];

const FAQItem = ({ question, answer, color, isOpen, onToggle }) => (
  <div className="bg-white rounded-xl border border-gray-200 shadow-sm w-full transition-shadow hover:shadow-md">
    <button
      onClick={onToggle}
      className="w-full flex justify-between items-start text-left p-6"
      aria-expanded={isOpen}
    >
      <div className="flex items-start space-x-4">
        <div className={`w-4 h-4 rounded-full mt-1 flex-shrink-0 ${color}`}></div>
        <h3 className="font-bold text-gray-800 text-lg">{question}</h3>
      </div>
      <BsChevronDown
        className={`w-5 h-5 text-gray-500 transform transition-transform duration-300 flex-shrink-0 ${
          isOpen ? "rotate-180" : "rotate-0"
        }`}
      />
    </button>
    <div
      className={`overflow-hidden transition-all duration-300 ease-in-out ${
        isOpen ? "max-h-60" : "max-h-0"
      }`}
    >
      <p className="text-gray-600 px-6 pb-6 pl-14">{answer}</p>
    </div>
  </div>
);

const FAQs = () => {
  const [openIndex, setOpenIndex] = useState(0);

  const handleToggle = (index) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <>
      <Header /> {/* ✅ Injected Header */}
      <section className="pt-8">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-extrabold text-gray-900 mb-12">
            Frequently Asked Questions
          </h2>
          <div className="space-y-4 text-left">
            {faqData.map((faq, index) => (
              <FAQItem
                key={index}
                {...faq}
                isOpen={openIndex === index}
                onToggle={() => handleToggle(index)}
              />
            ))}
          </div>
        </div>
      </section>
    </>
  );
};

export default FAQs;
