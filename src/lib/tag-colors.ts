// Deterministic color palette for tags — each tag gets a consistent, vibrant color
const TAG_COLORS = [
    { bg: "bg-rose-100 dark:bg-rose-900/30", text: "text-rose-700 dark:text-rose-400", activeBg: "bg-rose-500", border: "border-rose-300 dark:border-rose-700" },
    { bg: "bg-sky-100 dark:bg-sky-900/30", text: "text-sky-700 dark:text-sky-400", activeBg: "bg-sky-500", border: "border-sky-300 dark:border-sky-700" },
    { bg: "bg-amber-100 dark:bg-amber-900/30", text: "text-amber-700 dark:text-amber-400", activeBg: "bg-amber-500", border: "border-amber-300 dark:border-amber-700" },
    { bg: "bg-emerald-100 dark:bg-emerald-900/30", text: "text-emerald-700 dark:text-emerald-400", activeBg: "bg-emerald-500", border: "border-emerald-300 dark:border-emerald-700" },
    { bg: "bg-violet-100 dark:bg-violet-900/30", text: "text-violet-700 dark:text-violet-400", activeBg: "bg-violet-500", border: "border-violet-300 dark:border-violet-700" },
    { bg: "bg-cyan-100 dark:bg-cyan-900/30", text: "text-cyan-700 dark:text-cyan-400", activeBg: "bg-cyan-500", border: "border-cyan-300 dark:border-cyan-700" },
    { bg: "bg-pink-100 dark:bg-pink-900/30", text: "text-pink-700 dark:text-pink-400", activeBg: "bg-pink-500", border: "border-pink-300 dark:border-pink-700" },
    { bg: "bg-teal-100 dark:bg-teal-900/30", text: "text-teal-700 dark:text-teal-400", activeBg: "bg-teal-500", border: "border-teal-300 dark:border-teal-700" },
    { bg: "bg-orange-100 dark:bg-orange-900/30", text: "text-orange-700 dark:text-orange-400", activeBg: "bg-orange-500", border: "border-orange-300 dark:border-orange-700" },
    { bg: "bg-indigo-100 dark:bg-indigo-900/30", text: "text-indigo-700 dark:text-indigo-400", activeBg: "bg-indigo-500", border: "border-indigo-300 dark:border-indigo-700" },
    { bg: "bg-lime-100 dark:bg-lime-900/30", text: "text-lime-700 dark:text-lime-400", activeBg: "bg-lime-600", border: "border-lime-300 dark:border-lime-700" },
    { bg: "bg-fuchsia-100 dark:bg-fuchsia-900/30", text: "text-fuchsia-700 dark:text-fuchsia-400", activeBg: "bg-fuchsia-500", border: "border-fuchsia-300 dark:border-fuchsia-700" },
];

function hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) - hash) + str.charCodeAt(i);
        hash |= 0;
    }
    return Math.abs(hash);
}

export function getTagColor(tag: string) {
    return TAG_COLORS[hashString(tag) % TAG_COLORS.length];
}

/** Returns class string for a tag pill */
export function tagPillClass(tag: string, isActive: boolean): string {
    const c = getTagColor(tag);
    if (isActive) {
        return `${c.activeBg} text-white shadow-sm`;
    }
    return `${c.bg} ${c.text} hover:opacity-80`;
}

/** Returns class string for an inline tag badge (read-only, on cards etc) */
export function tagBadgeClass(tag: string): string {
    const c = getTagColor(tag);
    return `${c.bg} ${c.text}`;
}
