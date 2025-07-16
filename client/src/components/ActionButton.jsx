import React from 'react';

function ActionButton({ icon, title, subtitle, onClick }) {
    return (
        <div onClick={onClick} className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 text-left flex items-center gap-4 cursor-pointer hover:shadow-md transition-shadow">
            {icon}
            <div>
                <h6 className="font-bold text-slate-800">{title}</h6>
                <p className="text-slate-500 text-sm"><small>{subtitle}</small></p>
            </div>
        </div>
    );
}

export default ActionButton;