import React from 'react';
import { FaFacebookF, FaShareAlt, FaLinkedinIn, FaInstagram } from 'react-icons/fa';

const SocialIcon: React.FC<{ icon: React.ReactNode; bgColor: string; href?: string }> = ({ icon, bgColor, href = '#' }) => (
  <a href={href} target="_blank" rel="noopener noreferrer" className={`w-12 h-12 rounded-full flex items-center justify-center text-white shadow-md hover:opacity-90 transition-opacity ${bgColor}`}>
    {icon}
  </a>
);

const TeamMember: React.FC<{ imgSrc: string; name: string; title: string }> = ({ imgSrc, name, title }) => (
  <div className="text-center">
    <img
      src={imgSrc}
      alt={name}
      className="w-32 h-32 rounded-full mx-auto mb-4 object-cover shadow-lg border-2 border-white"
    />
    <h4 className="font-bold text-lg text-gray-800">{name}</h4>
    <p className="text-gray-500">{title}</p>
  </div>
);

const Contact: React.FC = () => {
  return (
    <section className="py-16 sm:py-24">
      <div className="max-w-4xl mx-auto text-center">
        <h2 className="text-4xl md:text-5xl font-extrabold text-gray-900 mb-12">
          Contact
        </h2>

        {/* Simplified Contact Form */}
        <form className="max-w-2xl mx-auto mb-16" onSubmit={(e) => e.preventDefault()}>
            <div className="flex items-center bg-white p-2 rounded-full border border-gray-200 shadow-md focus-within:ring-2 focus-within:ring-blue-500 transition-all">
                <textarea
                    name="message"
                    rows={1}
                    className="flex-grow bg-transparent text-lg text-gray-700 placeholder-gray-500 px-4 py-2 resize-none focus:outline-none"
                    placeholder="Write your message..."
                ></textarea>
                <button
                    type="submit"
                    className="bg-blue-600 text-white font-semibold py-3 px-8 rounded-full hover:bg-blue-700 transition-all duration-300 flex-shrink-0"
                >
                    Send
                </button>
            </div>
        </form>

        {/* Social Media Icons */}
        <div className="flex justify-center items-center space-x-4 mb-20">
            <SocialIcon icon={<FaFacebookF size={20} />} bgColor="bg-blue-600" />
            <SocialIcon icon={<FaShareAlt size={20} />} bgColor="bg-blue-500" />
            <SocialIcon icon={<FaLinkedinIn size={20} />} bgColor="bg-blue-800" />
            <SocialIcon icon={<FaInstagram size={20} />} bgColor="bg-gradient-to-tr from-yellow-400 via-red-500 to-purple-500" />
        </div>

        {/* Our Team Section */}
        <div>
            <h3 className="text-3xl font-bold text-gray-900 mb-10">Our Team</h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-y-12 sm:gap-x-8">
                <TeamMember 
                    imgSrc="https://images.unsplash.com/photo-1508214751196-bcfd4ca60f91?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=256&q=80" 
                    name="Sarah Doe" 
                    title="Support Manager" 
                />
                <TeamMember 
                    imgSrc="https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=256&q=80" 
                    name="John Smith" 
                    title="Lead Developer" 
                />
                <TeamMember 
                    imgSrc="https://images.unsplash.com/photo-1494790108377-be9c29b29330?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=256&q=80" 
                    name="Emma Johnson" 
                    title="Product Designer" 
                />
            </div>
        </div>
      </div>
    </section>
  );
};

export default Contact;