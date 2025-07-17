import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { useAuth } from "../context/AuthContext";
import { Github } from "lucide-react";
import { FcGoogle } from "react-icons/fc";


export default function SignInCard() {
  const navigate = useNavigate();
  const { signIn, signInWithGoogle, signInWithGitHub } = useAuth();

  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });

  const { mutate, isPending, isError, error } = useMutation({
    mutationFn: signIn,
    onSuccess: () => navigate("/dashboard"),
  });

  function handleChange(e) {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  }

  function handleSubmit(e) {
    e.preventDefault();
    mutate(formData);
  }

  return (
    <div className="flex min-h-full flex-col justify-center px-4 py-8 sm:px-6 lg:px-8 w-full max-w-sm mx-auto bg-white rounded-xl shadow-md">
      <div className="text-center">
        <button onClick={() => navigate("/")}>
          <img alt="Your Company" src="/logo.png" className="h-20 w-auto mx-auto" />
        </button>
        <h2 className="mt-4 text-2xl font-bold tracking-tight text-gray-900">
          Sign in to your account
        </h2>
      </div>

      <div className="mt-8 bg-blue-50 p-6 rounded-xl">
        <form onSubmit={handleSubmit} className="space-y-6">
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
              className="mt-2 block w-full rounded-md border border-gray-300 px-3 py-2 text-base text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <div className="flex items-center justify-between">
              <label htmlFor="password" className="block text-sm font-medium text-gray-900">
                Password
              </label>
              <div className="text-sm">
                <a href="#" className="font-semibold text-blue-600 hover:text-blue-500">
                  Forgot password?
                </a>
              </div>
            </div>
            <input
              id="password"
              name="password"
              type="password"
              required
              value={formData.password}
              onChange={handleChange}
              className="mt-2 block w-full rounded-md border border-gray-300 px-3 py-2 text-base text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {isError && <p className="text-sm text-red-600">{error.message}</p>}

          <div>
            <button
              type="submit"
              disabled={isPending}
              className="w-full flex justify-center items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
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
          </div>
        </form>

        {/* Divider */}
        <div className="flex items-center my-6">
          <div className="flex-grow border-t border-gray-300" />
          <span className="mx-4 text-sm text-gray-500">OR</span>
          <div className="flex-grow border-t border-gray-300" />
        </div>

        {/* Social Buttons */}
        <div className="space-y-3">
          <button
            onClick={() => signInWithGoogle()}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-md border border-gray-300 bg-white text-sm font-semibold text-gray-700 hover:bg-gray-100"
          >
            <FcGoogle size={20} />
            Continue with Google
          </button>

          {/* <button
            onClick={() => signInWithGitHub()}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-md border border-gray-300 bg-gray-900 text-sm font-semibold text-white hover:bg-black"
          >
            <Github size={20} />
            Continue with GitHub
          </button> */}
        </div>

        <p className="mt-6 text-center text-sm text-gray-500">
          Not a member?{" "}
          <button
            onClick={() => navigate("/signup")}
            className="font-semibold text-blue-600 hover:text-blue-500"
          >
            Sign up
          </button>
        </p>
      </div>
    </div>
  );
}
