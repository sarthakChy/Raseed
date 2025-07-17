import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import { useAuth } from "../context/AuthContext";

export default function SignUpCard() {
  const navigate = useNavigate();
  const { signUp } = useAuth();

  const [formData, setFormData] = useState({
    email: "",
    password: "",
    confirmPassword: "",
    name: "",
  });

  const { mutate, isPending, isError, error } = useMutation({
    mutationFn: signUp,
    onSuccess: () => navigate("/dashboard"),
  });

  function handleChange(e) {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  }

  function handleSubmit(e) {
    e.preventDefault();
    const { email, password, confirmPassword, name } = formData;

    if (password !== confirmPassword) {
      alert("Passwords do not match");
      return;
    }

    mutate({ email, password, name });
  }

  return (
    <div className="flex min-h-full flex-col justify-center px-4 py-8 sm:px-6 lg:px-8 w-full max-w-sm mx-auto bg-white rounded-xl shadow-md">
      <div className="text-center">
        <button onClick={() => navigate("/")}>
          <img alt="Your Company" src="/logo.png" className="h-20 w-auto mx-auto" />
        </button>
        <h2 className="mt-4 text-2xl font-bold tracking-tight text-gray-900">
          Create your Florette account
        </h2>
      </div>

      <div className="mt-8 bg-blue-50 p-6 rounded-xl">
        <form onSubmit={handleSubmit} className="space-y-3">
          {["name", "email", "password", "confirmPassword"].map((field) => (
            <div key={field}>
              <label
                htmlFor={field}
                className="block text-sm font-medium text-gray-900 capitalize"
              >
                {field === "confirmPassword" ? "Confirm Password" : field}
              </label>
              <input
                id={field}
                name={field}
                type={
                  field === "password" || field === "confirmPassword"
                    ? "password"
                    : "text"
                }
                required
                autoComplete={field}
                value={formData[field]}
                onChange={handleChange}
                className="mt-2 block w-full rounded-md border border-gray-300 px-3 py-2 text-base text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          ))}

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
              <span>Sign up</span>
            </button>
          </div>
        </form>

        <p className="mt-6 text-center text-sm text-gray-500">
          Already a member?{" "}
          <button
            onClick={() => navigate("/signin")}
            className="font-semibold text-blue-600 hover:text-blue-500"
          >
            Sign in
          </button>
        </p>
      </div>
    </div>
  );
}
