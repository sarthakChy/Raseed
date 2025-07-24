import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { useAuth } from "../context/AuthContext";
import { FcGoogle } from "react-icons/fc";

export default function SignInCard() {
  const navigate = useNavigate();
  const { signIn, signInWithGoogle } = useAuth();

  const [formData, setFormData] = useState({ email: "", password: "" });

  const { mutate, isPending, isError, error } = useMutation({
    mutationFn: signIn,
    onSuccess: () => navigate("/getstarted"),
  });

  function handleChange(e) {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  }

  function handleSubmit(e) {
    e.preventDefault();
    mutate(formData);
  }

<<<<<<< HEAD
=======
<<<<<<< HEAD
  async function handleGoogle() {
    try {
      await signInWithGoogle();
      navigate('/getstarted'); 
    } catch (err) {
      console.error("Google sign-in failed:", err);
    }
  }


>>>>>>> 9f4e6ef (I have no idea what is this at this point)
  return (
      <div className="w-full max-w-sm bg-white p-6 sm:p-5 rounded-xl shadow-md">
        <div className="text-center">
          <button onClick={() => navigate("/")}>
            <img src="/raseed-logo.png" alt="Logo" className="h-16 mx-auto" />
          </button>
          <h2 className="mt-3 text-xl font-bold text-gray-900">
            Sign in to your account
          </h2>
        </div>

  return (
      <div className="w-full max-w-sm bg-white p-6 sm:p-5 rounded-xl shadow-md">
        <div className="text-center">
          <button onClick={() => navigate("/")}>
            <img src="/raseed-logo.png" alt="Logo" className="h-16 mx-auto" />
          </button>
          <h2 className="mt-3 text-xl font-bold text-gray-900">
            Sign in to your account
          </h2>
        </div>

        <form onSubmit={handleSubmit} className="mt-5 space-y-4">
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-900">
              Email address
            </label>
            <input
              id="email"
              name="email"
              type="email"
              required
              value={formData.email}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-1.5 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[#4285F4]"
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-900">
              Password
            </label>
            <input
              id="password"
              name="password"
              type="password"
              required
              value={formData.password}
              onChange={handleChange}
              className="mt-1 block w-full rounded-md border border-gray-300 px-3 py-1.5 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[#4285F4]"
            />
            <div className="mt-1 text-right text-xs">
              <a href="#" className="text-[#DB4437] hover:underline">
                Forgot password?
              </a>
            </div>
          </div>

          {isError && (
            <p className="text-xs text-[#DB4437]">
              {error.message}
            </p>
          )}

          <button
            type="submit"
            disabled={isPending}
            className="w-full flex items-center justify-center gap-2 rounded-md bg-[#4285F4] px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-[#3367D6] focus:outline-none focus:ring-2 focus:ring-[#4285F4]"
          >
            {isPending && (
              <svg
                className="animate-spin h-4 w-4 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                />
              </svg>
            )}
            <span>Sign in</span>
          </button>
        </form>

        {/* Divider */}
        <div className="flex items-center my-4">
          <div className="flex-grow border-t border-gray-300" />
          <span className="mx-2 text-xs text-gray-500">OR</span>
          <div className="flex-grow border-t border-gray-300" />
        </div>

        {/* Google Auth */}
        <button
          onClick={handleGoogle}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-md border border-gray-300 bg-white text-sm font-semibold text-gray-700 hover:bg-gray-100"
        >
          <FcGoogle size={20} />
          Continue with Google
        </button>

        <p className="mt-4 text-center text-xs text-gray-500">
          Not a member?{" "}
          <button
            onClick={() => navigate("/signup")}
            className="font-semibold text-[#0F9D58] hover:text-green-600"
          >
            Sign up
          </button>
        </p>
      </div>
  );
}
