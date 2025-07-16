import React from 'react';
import { Camera, MessageSquare } from 'lucide-react';
import ActionButton from './ActionButton';
import { PAGES } from '../constants/pages';

function QuickActions({ onNavigate }) {
    return (
        <section>
            <h4 className="font-bold text-xl text-slate-700 mb-3">Quick Actions</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-1 gap-4">
                <ActionButton
                    icon={<Camera className="h-7 w-7 text-blue-600" />}
                    title="Capture Receipt"
                    subtitle="Photo • Video • Stream"
                    onClick={() => onNavigate(PAGES.CAPTURE_RECEIPT)}
                />
                <ActionButton
                    icon={<MessageSquare className="h-7 w-7 text-purple-600" />}
                    title="Ask RASEED"
                    subtitle="Any language supported"
                    onClick={() => onNavigate(PAGES.ASK_RASEED)}
                />
            </div>
        </section>
    );
}

export default QuickActions;