import React from "react";
import {
  FaFacebookF,
  FaShareAlt,
  FaLinkedinIn,
  FaInstagram,
} from "react-icons/fa";
import Header from "../components/Header"; // Adjust the path as needed

// Social Icon Button
const SocialIcon = ({ icon, bgColor, href = "#" }) => (
  <a
    href={href}
    target="_blank"
    rel="noopener noreferrer"
    className={`w-12 h-12 rounded-full flex items-center justify-center text-white shadow-md hover:scale-105 transition-transform duration-200 ${bgColor}`}
    aria-label="Social media link"
  >
    {icon}
  </a>
);

// Team Member Card
const TeamMember = ({ imgSrc, name, title }) => (
  <div className="text-center">
    <img
      src={imgSrc}
      alt={name}
      className="w-32 h-32 rounded-full mx-auto mb-4 object-cover shadow-lg border-2 border-white"
    />
    <h4 className="font-semibold text-lg text-gray-800">{name}</h4>
    <p className="text-gray-500">{title}</p>
  </div>
);

// Main Contact Page
const Contact = () => {
  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Form submitted");
  };

  return (
    <>
      <Header /> {/* ⬅ Add your shared navigation/header here */}
      <section className="px-4 py-4 max-w-6xl mx-auto">
        {/* Page Title */}
        <h2 className="text-4xl md:text-5xl font-extrabold text-center text-gray-900 mb-10">
          Get in Touch
        </h2>

        {/* Contact Form */}
        <form onSubmit={handleSubmit} className="max-w-2xl mx-auto mb-12">
          <div className="flex items-center bg-white p-2 rounded-full border border-gray-300 shadow-sm focus-within:ring-2 focus-within:ring-blue-500">
            <textarea
              name="message"
              rows={1}
              required
              className="flex-grow bg-transparent text-base text-gray-700 placeholder-gray-500 px-4 py-2 resize-none focus:outline-none"
              placeholder="Write your message..."
            ></textarea>
            <button
              type="submit"
              className="bg-blue-600 text-white font-medium py-3 px-8 rounded-full hover:bg-blue-700 transition-all duration-300"
            >
              Send
            </button>
          </div>
        </form>

        {/* Social Media Links */}
        <div className="flex justify-center gap-4 mb-16">
          <SocialIcon icon={<FaFacebookF size={20} />} bgColor="bg-blue-600" />
          <SocialIcon icon={<FaShareAlt size={20} />} bgColor="bg-blue-500" />
          <SocialIcon icon={<FaLinkedinIn size={20} />} bgColor="bg-blue-800" />
          <SocialIcon
            icon={<FaInstagram size={20} />}
            bgColor="bg-gradient-to-tr from-yellow-400 via-red-500 to-purple-500"
          />
        </div>

        {/* Our Team Section */}
        <div>
          <h3 className="text-3xl font-bold text-center text-gray-900 mb-10">
            Our Team
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-y-12 sm:gap-x-8">
            <TeamMember
              imgSrc="https://images.unsplash.com/photo-1508214751196-bcfd4ca60f91?ixlib=rb-4.0.3&auto=format&fit=crop&w=256&q=80"
              name="Sarah Doe"
              title="Support Manager"
            />
            <TeamMember
              imgSrc="https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?ixlib=rb-4.0.3&auto=format&fit=crop&w=256&q=80"
              name="John Smith"
              title="Lead Developer"
            />
            <TeamMember
              imgSrc="https://images.unsplash.com/photo-1494790108377-be9c29b29330?ixlib=rb-4.0.3&auto=format&fit=crop&w=256&q=80"
              name="Emma Johnson"
              title="Product Designer"
            />
          </div>
        </div>
      </section>
    </>
  );
};

export default Contact;
