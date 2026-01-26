import { Header } from "../../components/custom/header";

export const Webapps = () => {
  return (
    <div className="h-full flex flex-col">
      <Header />
      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-4">Web Apps</h1>
          <p className="text-gray-600 dark:text-gray-400">Coming soon...</p>
        </div>
      </main>
    </div>
  );
};
