import React, { useEffect, useState, useRef } from "react";
import { useAuth } from "../context/AuthContext";
import { useNavigate, Link } from "react-router-dom";

const Header = () => {
  const logo = "raseed-logo.png";
  const dummyAvatar = "https://i.pravatar.cc/300";
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const dropdownRef = useRef(null);
  const mobileMenuRef = useRef(null);

  useEffect(() => {
    setIsAuthenticated(!!user);
  }, [user]);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Close mobile menu on outside click
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (
        mobileMenuOpen &&
        mobileMenuRef.current &&
        !mobileMenuRef.current.contains(e.target)
      ) {
        setMobileMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [mobileMenuOpen]);

  const NavLink = ({ to, children, onClick }) => (
    <Link
      to={to}
      onClick={() => {
        onClick?.();
        setMobileMenuOpen(false);
      }}
      className="text-gray-700 hover:text-black transition-colors duration-300 text-base font-medium"
    >
      {children}
    </Link>
  );

  return (
    <header className="py-6 relative z-50">
      <nav className="flex justify-between items-center px-4">
        {/* Logo */}
        <div className="text-2xl font-extrabold tracking-tight">
          <div className="flex items-center space-x-2">
            <img
              src={logo}
              alt="Raseed Logo"
              className="w-8 h-8 object-contain"
            />
            <span className="text-2xl font-bold text-gray-800">RASEED</span>
          </div>
        </div>

        {/* Desktop Links */}
        <div className="hidden md:flex items-center space-x-8">
          <NavLink to="/">Home</NavLink>
          <NavLink to="/about">About</NavLink>
          <NavLink to="/getstarted">Get Started</NavLink>
          <NavLink to="/upgrade">Upgrade Plan</NavLink>
          <NavLink to="/faq">FAQs</NavLink>
          <NavLink to="/contact">Contact</NavLink>

          {isAuthenticated && (
            <div className="relative" ref={dropdownRef}>
              <button
                onClick={() => setDropdownOpen((prev) => !prev)}
                className="w-10 h-10 rounded-full overflow-hidden border-2 border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400"
              >
                <img
                  src={dummyAvatar}
                  alt="User Avatar"
                  className="w-full h-full object-cover"
                />
              </button>

              {dropdownOpen && (
                <div className="absolute right-0 mt-2 w-40 bg-white border border-gray-200 rounded-lg shadow-lg z-50">
                  <button
                    className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 rounded-lg"
                    onClick={() => {
                      setDropdownOpen(false);
                      alert("Option 1 clicked");
                    }}
                  >
                    Option 1
                  </button>
                  <button
                    className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 rounded-lg"
                    onClick={() => {
                      setDropdownOpen(false);
                      alert("Option 2 clicked");
                    }}
                  >
                    Option 2
                  </button>
                  <button
                    onClick={() => {
                      logout();
                      navigate("/");
                    }}
                    className="block w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-600 hover:text-white rounded-lg"
                  >
                    Logout
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Hamburger for mobile */}
        <div className="md:hidden">
          <button onClick={() => setMobileMenuOpen(true)}>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-7 w-7 text-gray-800"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 6h16M4 12h16m-7 6h7"
              />
            </svg>
          </button>
        </div>
      </nav>

      {/* Mobile Full-Screen Drawer */}
      <div
        className={`fixed inset-0 bg-white z-40 transform transition-transform duration-300 ease-in-out ${
          mobileMenuOpen ? "translate-y-0" : "-translate-y-full"
        }`}
      >
        <div
          ref={mobileMenuRef}
          className="flex flex-col items-center justify-start px-6 py-8 space-y-6 h-full"
        >
          <div className="flex justify-between items-center w-full">
            <div className="flex items-center space-x-2">
              <img
                src={logo}
                alt="Raseed Logo"
                className="w-8 h-8 object-contain"
              />
              <span className="text-xl font-bold text-gray-800">RASEED</span>
            </div>
            <button onClick={() => setMobileMenuOpen(false)}>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-6 w-6 text-gray-800"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>

          <NavLink to="/" onClick={() => setMobileMenuOpen(false)}>
            Home
          </NavLink>
          <NavLink to="/about">About</NavLink>
          <NavLink to="/getstarted">Get Started</NavLink>
          <NavLink to="/upgrade">Upgrade Plan</NavLink>
          <NavLink to="/faq">FAQs</NavLink>
          <NavLink to="/contact">Contact</NavLink>

          {isAuthenticated && (
            <>
              {/* <button
                onClick={() => alert("Option 1 clicked")}
                className="text-gray-700 hover:text-black"
              >
                Option 1
              </button>
              <button
                onClick={() => alert("Option 2 clicked")}
                className="text-gray-700 hover:text-black"
              >
                Option 2
              </button> */}
              <button
                onClick={() => {
                  logout();
                  navigate("/");
                }}
                className="text-red-600 hover:text-white hover:bg-red-500 px-4 font-bold rounded"
              >
                Logout
              </button>
            </>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;
