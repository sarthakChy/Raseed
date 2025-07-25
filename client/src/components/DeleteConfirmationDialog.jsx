import React from "react";

const DeleteConfirmationDialog = ({ isOpen, onConfirm, onCancel }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-sm p-6 text-center space-y-4">
        <h2 className="text-xl font-semibold text-gray-800">Delete this receipt?</h2>
        <p className="text-gray-600 text-sm">This action cannot be undone.</p>
        <div className="flex justify-center space-x-4 pt-4">
          <button
            className="px-4 py-2 bg-gray-200 rounded-lg hover:bg-gray-300 text-gray-800 font-medium"
            onClick={onCancel}
          >
            Cancel
          </button>
          <button
            className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 font-medium"
            onClick={onConfirm}
          >
            Delete
          </button>
        </div>
      </div>
    </div>
  );
};

export default DeleteConfirmationDialog;
